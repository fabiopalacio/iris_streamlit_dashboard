import pickle

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt


def plot_scatter_graph(df, df_input, x, y):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Dados de treinamento
    for species in df["Espécie"].unique():
        dados = df[df["Espécie"] == species]

        ax.scatter(
            dados[x],
            dados[y],
            label=species,
            alpha=0.7
        )

    # Ponto inserido pelo usuário
    ax.scatter(
        df_input[x],
        df_input[y],
        color="red",
        marker="*",
        s=250,
        edgecolors="black",
        label="Usuário",
        zorder=5
    )

    ax.set_xlabel(f"{x} (cm)")
    ax.set_ylabel(f"{y} (cm)")
    ax.set_title("Medidas de flores do gênero Iris")
    ax.legend()
    ax.patch.set_facecolor('#f1fff7')
    fig.patch.set_facecolor('#f1fff7')
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)


def convert_data_to_dataframe(iris):
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    df.rename(columns={
        "sepal length (cm)": "Comprimento da sépala",
        "sepal width (cm)": "Largura da sépala",
        "petal length (cm)": "Comprimento da pétala",
        "petal width (cm)": "Largura da pétala"},
        inplace=True)
    df["Espécie"] = iris.target_names[iris.target]
    df["Espécie"] = "Iris " + df["Espécie"]
    return df


def write_user_data_as_table(df):
    userInputs = pd.DataFrame(
        {
            "Comprimento": {
                "Pétala": df["Comprimento da pétala"][0],
                "Sépala": df["Comprimento da sépala"][0],
            },
            "Largura": {
                "Pétala": df["Largura da pétala"][0],
                "Sépala": df["Largura da sépala"][0],
            },
        }
    )

    return userInputs


def make_prediction(df):
    load_clf = pickle.load(open("./model/iris_rfc.pkl", "rb"))

    # ---- Aplicação do classificador aos dados inseridos ----
    prediction = load_clf.predict(df.values)
    prediction_proba = load_clf.predict_proba(df.values)

    return (prediction, prediction_proba)


def set_title(level, align, text):
    return f"<{level} style='text-align:{align};'>{text}</{level}>"
