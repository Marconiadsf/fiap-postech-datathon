# Modelo Preditivo (Pergunta 9)
from __future__ import annotations

import argparse
import datetime as dt
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ======================================================================
# 0) CONFIGURAÇÃO GERAL
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

URL_PLANILHA = (
    "https://docs.google.com/spreadsheets/d/"
    "1td91KoeSgXrUrCVOUkLmONG9Go3LVcXpcNEw_XrL2R0/export?format=xlsx"
)

# Abas disponíveis na planilha
ABAS_POR_ANO: dict[int, str] = {
    2022: "PEDE2022",
    2023: "PEDE2023",
    2024: "PEDE2024",
}

_EPOCA_EXCEL = pd.Timestamp("1899-12-31")


# ======================================================================
# 1) UTILITÁRIOS DE PARSING (equivalentes aos do antigo pede_cleaning.py)
# ======================================================================

def _sem_bom(texto: Any) -> str:
    return str(texto).replace("\ufeff", "").strip()


def numero_pt_br_para_float(valor: Any) -> float:
    """Converte string numérica pt-BR (vírgula decimal) em float; vazio vira NaN."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return np.nan
    if isinstance(valor, (int, float, np.floating)) and not isinstance(valor, bool):
        return float(valor)
    texto = _sem_bom(valor)
    if texto == "" or texto.lower() in {"nan", "none", "na", "n/a"}:
        return np.nan
    texto = texto.replace('"', "").replace("'", "")
    texto = (
        texto.replace(".", "").replace(",", ".")
        if re.search(r",\d+$", texto)
        else texto.replace(",", ".")
    )
    try:
        return float(texto)
    except ValueError:
        return np.nan


def _unir_colunas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    """Funde colunas com o mesmo nome base que o pandas separou com sufixo '.1'."""
    out = df.copy()
    if "Destaque IPV.1" in out.columns and "Destaque IPV" in out.columns:
        out["Destaque IPV"] = out["Destaque IPV"].where(
            out["Destaque IPV"].notna() & (out["Destaque IPV"].astype(str).str.strip() != ""),
            out["Destaque IPV.1"],
        )
        out = out.drop(columns=["Destaque IPV.1"])
    if "Ativo/ Inativo.1" in out.columns and "Ativo/ Inativo" in out.columns:
        out["Ativo/ Inativo"] = out["Ativo/ Inativo"].where(
            out["Ativo/ Inativo"].notna() & (out["Ativo/ Inativo"].astype(str).str.strip() != ""),
            out["Ativo/ Inativo.1"],
        )
        out = out.drop(columns=["Ativo/ Inativo.1"])
    return out


def _normalizar_nome_coluna(nome: str) -> str:
    return (
        _sem_bom(nome)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("º", "o")
        .replace("°", "o")
    )


def _celula_para_texto(valor: Any) -> str:
    """
    Uniformiza uma célula lida do Excel para o mesmo formato que o pandas
    produzia ao ler o CSV original (dtype=str, keep_default_na=False):
      - float "inteiro" (ex.: 7.0) -> '7'
      - demais valores -> str(valor).strip()
    """
    if valor is None:
        return ""
    if isinstance(valor, float) and np.isnan(valor):
        return ""
    if isinstance(valor, (pd.Timestamp, dt.datetime, dt.date)):
        numero = float((pd.Timestamp(valor) - _EPOCA_EXCEL).days)
        return _celula_para_texto(numero)
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))
    return str(valor).strip()


def carregar_aba_planilha(url: str, aba: str) -> pd.DataFrame:   
    bruto = pd.read_excel(url, sheet_name=aba)
    bruto = _unir_colunas_duplicadas(bruto)
    bruto.columns = [_normalizar_nome_coluna(c) for c in bruto.columns]

    for coluna in bruto.columns:
        if coluna == "data_de_nasc":
            continue
        bruto[coluna] = bruto[coluna].map(_celula_para_texto)

    return bruto


# ======================================================================
# 2) NORMALIZAÇÃO DA COLUNA 'fase' 
# ======================================================================

def normalizar_rotulo_fase(valor: Any) -> float:
    # Converte rótulos brutos de fase em escalar 0..8 ou NaN.
    if valor is None:
        return np.nan
    if isinstance(valor, (bool, np.bool_)):
        return np.nan
    if isinstance(valor, (int, np.integer)):
        v = int(valor)
        return float(v) if 0 <= v <= 8 else np.nan
    if isinstance(valor, (float, np.floating)):
        if np.isnan(valor):
            return np.nan
        v = int(round(float(valor)))
        return float(v) if 0 <= v <= 8 else np.nan

    texto = _sem_bom(valor)
    if texto == "" or texto.lower() in {"nan", "none", "na", "n/a"}:
        return np.nan

    def _dobra_acentos(t: str) -> str:
        t = unicodedata.normalize("NFD", t)
        return "".join(ch for ch in t if unicodedata.category(ch) != "Mn").lower()

    bruto = str(texto).strip()
    if _dobra_acentos(bruto) in {"alfa", "alpha"}:
        return 0.0

    sem_prefixo = re.sub(r"^fase\s*", "", bruto, flags=re.IGNORECASE).strip()
    if _dobra_acentos(sem_prefixo) in {"alfa", "alpha"}:
        return 0.0

    if re.fullmatch(r"[0-8]", sem_prefixo):
        return float(sem_prefixo)

    combinado_letras = re.fullmatch(r"([0-8])(\D+)", sem_prefixo)
    if combinado_letras:
        return float(int(combinado_letras.group(1)))

    combinado_sufixo = re.match(r"^([0-8])(.+)$", sem_prefixo)
    if combinado_sufixo:
        digito = int(combinado_sufixo.group(1))
        resto = combinado_sufixo.group(2)
        if re.search(r"\d", resto):
            return np.nan
        return float(digito)

    return np.nan


def fase_para_inteiro(serie: pd.Series) -> pd.Series:
    mapeada = serie.map(normalizar_rotulo_fase)
    valores: list[int | None] = []
    for v in mapeada.tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            valores.append(None)
        else:
            inteiro = int(round(float(v)))
            valores.append(inteiro if 0 <= inteiro <= 8 else None)
    return pd.Series(valores, index=serie.index, dtype="Int64")


def normalizar_coluna_fase(df: pd.DataFrame) -> pd.DataFrame:
   # Garante 'fase' como inteiro 0-8 (nullable)
    if "fase" not in df.columns:
        return df
    out = df.copy()
    out["fase"] = fase_para_inteiro(out["fase"])
    return out


def _interpretar_genero(valor: Any) -> str:
    texto = _sem_bom(valor).lower()
    if texto in {"menina", "feminino", "f"}:
        return "Feminino"
    if texto in {"menino", "masculino", "m"}:
        return "Masculino"
    if texto == "":
        return ""
    return _sem_bom(valor)


# ======================================================================
# 3) HARMONIZAÇÃO POR ANO
# ======================================================================

def _harmonizar_2022(df: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame()
    h["ra"] = df["ra"].astype(str).str.strip()
    h["ano_cohorte"] = 2022
    h["fase"] = df["fase"].astype(str).str.strip()
    h["turma"] = df.get("turma", "")
    h["nome"] = df.get("nome", "")
    h["ano_nascimento"] = df.get("ano_nasc", "").map(numero_pt_br_para_float)
    h["data_nasc"] = pd.NaT
    h["idade_referencia"] = df.get("idade_22", "").map(numero_pt_br_para_float)
    h["genero"] = df["gênero"].map(_interpretar_genero)
    h["ano_ingresso"] = df.get("ano_ingresso", "").map(numero_pt_br_para_float)
    h["instituicao_ensino"] = df["instituição_de_ensino"]
    h["escola"] = ""
    h["status_aluno"] = ""
    for p in ["pedra_20", "pedra_21", "pedra_22"]:
        h[p] = df.get(p, "")
    h["pedra_23"] = ""
    h["pedra_atual"] = df.get("pedra_22", "")
    h["inde_cohorte"] = df.get("inde_22", "").map(numero_pt_br_para_float)
    h["inde_hist_22"] = df.get("inde_22", "").map(numero_pt_br_para_float)
    h["inde_hist_23"] = np.nan
    h["inde_hist_24"] = np.nan
    h["cg"] = df.get("cg", "").map(numero_pt_br_para_float)
    h["cf"] = df.get("cf", "").map(numero_pt_br_para_float)
    h["ct"] = df.get("ct", "").map(numero_pt_br_para_float)
    h["n_avaliacoes"] = df.get("no_av", "").map(numero_pt_br_para_float)
    for ind in ["iaa", "ieg", "ips", "ida", "ipv", "ian"]:
        h[ind] = df.get(ind, "").map(numero_pt_br_para_float)
    h["ipp"] = np.nan  # não consta no layout 2022 da planilha
    h["mat"] = df.get("matem", "").map(numero_pt_br_para_float)
    h["por"] = df.get("portug", "").map(numero_pt_br_para_float)
    h["ing"] = df["inglês"].map(numero_pt_br_para_float)
    h["indicado"] = df.get("indicado", "")
    h["atingiu_pv"] = df.get("atingiu_pv", "")
    h["fase_ideal"] = df.get("fase_ideal", "")
    h["defasagem"] = df.get("defas", "").map(numero_pt_br_para_float)
    h["rec_psicologia"] = df.get("rec_psicologia", "")
    h["destaque_ieg"] = df.get("destaque_ieg", "")
    h["destaque_ida"] = df.get("destaque_ida", "")
    h["destaque_ipv"] = df.get("destaque_ipv", "")
    return h


def _harmonizar_2023_2024(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    h = pd.DataFrame()
    h["ra"] = df["ra"].astype(str).str.strip()
    h["ano_cohorte"] = ano
    h["fase"] = df["fase"].astype(str).str.strip()
    h["turma"] = df.get("turma", "")
    h["nome"] = df.get("nome_anonimizado", "")
    h["data_nasc"] = pd.to_datetime(df.get("data_de_nasc", ""), errors="coerce", format="mixed")
    h["ano_nascimento"] = h["data_nasc"].dt.year
    h["idade_referencia"] = df.get("idade", "").map(numero_pt_br_para_float)
    h["genero"] = df["gênero"].map(_interpretar_genero)
    h["ano_ingresso"] = df.get("ano_ingresso", "").map(numero_pt_br_para_float)
    h["instituicao_ensino"] = df["instituição_de_ensino"]
    h["escola"] = df.get("escola", "") if ano == 2024 else ""
    h["status_aluno"] = df.get("ativo__inativo", "") if ano == 2024 else ""

    for p in ["pedra_20", "pedra_21", "pedra_22", "pedra_23"]:
        h[p] = df.get(p, "")
    h["pedra_atual"] = df.get(f"pedra_{ano}", "")

    col_inde = f"inde_{ano}"
    h["inde_cohorte"] = df.get(col_inde, "").map(numero_pt_br_para_float)
    h["inde_hist_22"] = df.get("inde_22", "").map(numero_pt_br_para_float)
    h["inde_hist_23"] = df.get("inde_23", "").map(numero_pt_br_para_float)
    h["inde_hist_24"] = (
        df.get("inde_24", "").map(numero_pt_br_para_float) if "inde_24" in df.columns else np.nan
    )

    h["cg"] = df.get("cg", "").map(numero_pt_br_para_float)
    h["cf"] = df.get("cf", "").map(numero_pt_br_para_float)
    h["ct"] = df.get("ct", "").map(numero_pt_br_para_float)
    h["n_avaliacoes"] = df.get("no_av", "").map(numero_pt_br_para_float)
    for ind in ["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]:
        h[ind] = df.get(ind, "").map(numero_pt_br_para_float)
    h["mat"] = df.get("mat", "").map(numero_pt_br_para_float)
    h["por"] = df.get("por", "").map(numero_pt_br_para_float)
    h["ing"] = df.get("ing", "").map(numero_pt_br_para_float)
    h["indicado"] = df.get("indicado", "")
    h["atingiu_pv"] = df.get("atingiu_pv", "")
    h["fase_ideal"] = df.get("fase_ideal", "")
    h["defasagem"] = df.get("defasagem", "").map(numero_pt_br_para_float)
    h["rec_psicologia"] = df.get("rec_psicologia", "")
    h["destaque_ieg"] = df.get("destaque_ieg", "")
    h["destaque_ida"] = df.get("destaque_ida", "")
    h["destaque_ipv"] = df.get("destaque_ipv", "")
    return h


def harmonizar_ano(df_bruto: pd.DataFrame, ano: int) -> pd.DataFrame:
    if ano == 2022:
        return _harmonizar_2022(df_bruto)
    if ano in (2023, 2024):
        return _harmonizar_2023_2024(df_bruto, ano)
    raise ValueError(f"Ano não suportado: {ano}")


# ======================================================================
# 4) UNIFICAÇÃO DAS 3 ABAS
# ======================================================================

def construir_base_unificada(
    url: str = URL_PLANILHA,
    abas: dict[int, str] | None = None,
    caminho_parquet: Path | None = None,
) -> pd.DataFrame:
    """Lê as 3 abas da planilha, harmoniza e empilha em um único dataframe."""
    abas = abas or ABAS_POR_ANO
    partes: list[pd.DataFrame] = []
    for ano, nome_aba in sorted(abas.items()):
        bruto = carregar_aba_planilha(url, nome_aba)
        partes.append(harmonizar_ano(bruto, ano))

    unificado = pd.concat(partes, ignore_index=True)
    unificado["fase"] = fase_para_inteiro(unificado["fase"])

    duplicados = unificado.duplicated(subset=["ra", "ano_cohorte"], keep=False)
    if duplicados.any():
        problemas = unificado.loc[duplicados, ["ra", "ano_cohorte"]].drop_duplicates()
        raise ValueError(f"Chaves duplicadas (ra, ano_cohorte):\n{problemas.head(20)}")

    if caminho_parquet is not None:
        caminho_parquet.parent.mkdir(parents=True, exist_ok=True)
        unificado.to_parquet(caminho_parquet, index=False)

    return unificado


def relatorio_limpeza(df: pd.DataFrame) -> dict[str, Any]:
    por_ano = df.groupby("ano_cohorte").size().to_dict()
    return {
        "n_linhas": len(df),
        "n_ra_distintos": df["ra"].nunique(),
        "linhas_por_ano": por_ano,
        "colunas": list(df.columns),
        "taxa_nulo_inde_cohorte": float(df["inde_cohorte"].isna().mean()),
        "taxa_nulo_ian": float(df["ian"].isna().mean()),
        "taxa_nulo_fase": float(df["fase"].isna().mean()),
    }


# ======================================================================
# 5) MODELO DE RISCO DE DEFASAGEM
# ======================================================================
FEATURES_NUMERICAS: list[str] = [
    "idade_referencia",
    "fase",
    "ano_ingresso",
    "cg",
    "cf",
    "ct",
    "n_avaliacoes",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "mat",
    "por",
    "ing",
    "ipv",
    "inde_hist_22",
    "inde_hist_23",
    "ano_cohorte",
]

FEATURES_CATEGORICAS: list[str] = ["genero"]


def montar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Monta X, y com alvo binário Defasagem < 0."""
    base = df.copy()
    base["fase"] = fase_para_inteiro(base["fase"])
    base["__alvo__"] = (base["defasagem"] < 0).astype(int)
    X = base[FEATURES_NUMERICAS + FEATURES_CATEGORICAS].copy()
    for coluna in FEATURES_CATEGORICAS:
        X[coluna] = X[coluna].astype(str).replace({"": "Desconhecido"})
    y = base["__alvo__"].values
    return X, y


def montar_pipeline() -> Pipeline:
    pre_processamento = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="median")),
                        ("escalonador", StandardScaler()),
                    ]
                ),
                FEATURES_NUMERICAS,
            ),
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputador", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                FEATURES_CATEGORICAS,
            ),
        ]
    )
    classificador = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre_processamento), ("clf", classificador)])


@dataclass
class ResultadoTreino:
    pipeline: Pipeline
    metricas: dict[str, Any]
    padroes_numericos: dict[str, float]
    padroes_categoricos: dict[str, str]


def treinar_modelo_risco(df: pd.DataFrame, tamanho_teste: float = 0.25, semente: int = 42) -> ResultadoTreino:
    X, y = montar_x_y(df)
    mascara = X[FEATURES_NUMERICAS].notna().any(axis=1)
    X, y = X.loc[mascara], y[mascara.values]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=tamanho_teste, random_state=semente, stratify=y
    )
    pipe = montar_pipeline()
    pipe.fit(X_treino, y_treino)
    proba = pipe.predict_proba(X_teste)[:, 1]
    predito = (proba >= 0.5).astype(int)
    metricas = {
        "roc_auc": float(roc_auc_score(y_teste, proba)),
        "classification_report": classification_report(y_teste, predito, digits=3),
        "n_train": int(len(X_treino)),
        "n_test": int(len(X_teste)),
        "positivos_rate": float(y.mean()),
    }
    padroes_numericos = X_treino[FEATURES_NUMERICAS].median(numeric_only=True).to_dict()
    padroes_categoricos = {
        c: (X_treino[c].mode().iloc[0] if len(X_treino[c].mode()) else "Desconhecido")
        for c in FEATURES_CATEGORICAS
    }
    return ResultadoTreino(
        pipeline=pipe,
        metricas=metricas,
        padroes_numericos=padroes_numericos,
        padroes_categoricos=padroes_categoricos,
    )


def salvar_pacote_modelo(caminho: Path, resultado: ResultadoTreino) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pacote = {
        "pipeline": resultado.pipeline,
        "numeric_features": FEATURES_NUMERICAS,
        "categorical_features": FEATURES_CATEGORICAS,
        "defaults_numeric": resultado.padroes_numericos,
        "defaults_cat": resultado.padroes_categoricos,
        "metrics": resultado.metricas,
        "target_definition": "P(defasagem < 0) — defasagem é o D (fase efetiva - ideal) na base harmonizada.",
    }
    joblib.dump(pacote, caminho)


def caminho_parquet_padrao(raiz: Path | None = None) -> Path:
    r = raiz or PROJECT_ROOT
    return r / "data_processed" / "pede_unificado.parquet"


def caminho_modelo_padrao(raiz: Path | None = None) -> Path:
    r = raiz or PROJECT_ROOT
    return r / "models" / "risk_defasagem.joblib"


def garantir_parquet(raiz: Path | None = None, url: str = URL_PLANILHA) -> Path:
    """Gera o parquet unificado a partir da planilha, se ainda não existir."""
    raiz = raiz or PROJECT_ROOT
    destino = caminho_parquet_padrao(raiz)
    if not destino.exists():
        construir_base_unificada(url=url, caminho_parquet=destino)
    return destino


def carregar_pacote_modelo(caminho: Path | None = None) -> dict[str, Any]:
    caminho = caminho or caminho_modelo_padrao()
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}. Rode: python model_training_pede.py --force"
        )
    return joblib.load(caminho)


def prever_probabilidade(pacote: dict[str, Any], linha: dict[str, Any]) -> float:
    """Retorna a probabilidade da classe positiva (Defasagem < 0)."""
    colunas = pacote["numeric_features"] + pacote["categorical_features"]
    X = pd.DataFrame([{k: linha.get(k, np.nan) for k in colunas}])
    return float(pacote["pipeline"].predict_proba(X)[0, 1])


def _linha_a_partir_dos_padroes(pacote: dict[str, Any]) -> dict[str, Any]:
    linha: dict[str, Any] = {}
    linha.update(dict(pacote.get("defaults_numeric") or {}))
    linha.update(dict(pacote.get("defaults_cat") or {}))
    for c in pacote.get("categorical_features") or []:
        if c in linha and linha[c] is not None and not isinstance(linha[c], str):
            linha[c] = str(linha[c])
    return linha


def _pacote_prediz_corretamente(pacote: dict[str, Any]) -> bool:
    """Detecta joblib treinado com outra versão do scikit-learn (evita quebrar em produção)."""
    try:
        prever_probabilidade(pacote, _linha_a_partir_dos_padroes(pacote))
        return True
    except Exception:
        return False


def garantir_modelo_salvo(raiz: Path | None = None, *, forcar: bool = False) -> Path:
    """
    Garante parquet + arquivo `risk_defasagem.joblib`.

    Usado pelo Streamlit na primeira execução. Com `forcar=True`, apaga o
    joblib existente e treina de novo.
    """
    raiz = raiz or PROJECT_ROOT
    garantir_parquet(raiz)
    destino = caminho_modelo_padrao(raiz)
    if destino.exists() and not forcar:
        try:
            pacote = joblib.load(destino)
            if (
                pacote.get("numeric_features") != FEATURES_NUMERICAS
                or pacote.get("categorical_features") != FEATURES_CATEGORICAS
            ):
                forcar = True
            elif not _pacote_prediz_corretamente(pacote):
                forcar = True
        except Exception:
            forcar = True
    if forcar and destino.exists():
        destino.unlink()
    if destino.exists() and not forcar:
        return destino
    df = pd.read_parquet(caminho_parquet_padrao(raiz))
    resultado = treinar_modelo_risco(df)
    salvar_pacote_modelo(destino, resultado)
    return destino


# ======================================================================
# 6) EXECUÇÃO VIA LINHA DE COMANDO
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpa a base PEDE (Google Sheets) e treina o modelo de risco de defasagem."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apaga o parquet e o joblib existentes e refaz tudo do zero.",
    )
    args = parser.parse_args()

    if args.force:
        for caminho in (caminho_parquet_padrao(PROJECT_ROOT), caminho_modelo_padrao(PROJECT_ROOT)):
            if caminho.exists():
                caminho.unlink()

    destino = garantir_modelo_salvo(PROJECT_ROOT, forcar=args.force)
    pacote = carregar_pacote_modelo(destino)

    print("Modelo disponível em:", destino)
    print("ROC-AUC (holdout):", pacote["metrics"].get("roc_auc"))


if __name__ == "__main__":
    main()