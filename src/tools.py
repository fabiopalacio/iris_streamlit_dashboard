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
