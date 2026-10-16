import streamlit as st
import pandas as pd
from sklearn import datasets
import pickle
from src.tools import convert_data_to_dataframe, plot_scatter_graph

st.set_page_config(
    page_title="Predição Iris",
    layout="wide"
)

iris = datasets.load_iris()


# --- Instruções da aplicação como expander ----
with st.expander("Sobre a aplicação"):
    st.markdown("<p>Aplicação que utiliza um modelo de <b>RandomForest</b> para prever a qual espécie do gênero de <i>Iris</i> as medidas inseridas pelo usuário pertence.</p>"
                "<p>As medidas obtidas pelo usuário devem ser inseridas no menu lateral.", True)
st.markdown("---")

# ---- Chamada da função para barra lateral ----
st.sidebar.header("Predição da espécie de Iris", text_alignment="center")
sepal_length = st.sidebar.slider("Comprimento da sépala", 4.3, 7.9, 5.4)
sepal_width = st.sidebar.slider("Largura da sépala", 2.0, 4.4, 3.4)
petal_length = st.sidebar.slider("Comprimento da pétala", 1.0, 6.9, 1.3)
petal_width = st.sidebar.slider("Largura da pétala", 0.1, 2.5, 0.2)
data = {
    "Comprimento da sépala": sepal_length,
    "Largura da sépala": sepal_width,
    "Comprimento da pétala": petal_length,
    "Largura da pétala": petal_width
}
df_input = pd.DataFrame(data, index=[0])


# ---- Criação de um novo DataFrame para exibição dos valores inseridos ----
# Foi criado um novo DataFrame para não interferir no dataFrame esperado pela análise
userInputs = pd.DataFrame(
    {
        "Comprimento": {
            "Pétala": df_input["Comprimento da pétala"][0],
            "Sépala": df_input["Comprimento da sépala"][0],
        },
        "Largura": {
            "Pétala": df_input["Largura da pétala"][0],
            "Sépala": df_input["Largura da sépala"][0],
        },
    }
)

# Reads the classification model
load_clf = pickle.load(open("./model/iris_rfc.pkl", "rb"))

# ---- Aplicação do classificador aos dados inseridos ----
prediction = load_clf.predict(df_input.values)
prediction_proba = load_clf.predict_proba(df_input.values)


# ---- Criação das colunas com os dados inseridos (esquerda) e resultado do algoritmo (direita) -----
left_column, right_column = st.columns(2)
with left_column:
    with st.container():
        st.markdown(
            "<h3 style='text-align:center;'>Parâmetros inseridos</h3>", True)
        st.write(userInputs)

with right_column:
    with st.container():
        st.markdown(
            "<h3 style='text-align:center;'>Espécie encontrada:</h3>", True)
        species_predicted = "Iris " + (
            str(iris.target_names[prediction])
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
        )

        st.markdown(
            "<div style='display: flex; justify-content:center;'><span style='font-style: italic;font-size:25px; font-weight: bold;color:red;'>"
            + species_predicted
            + "</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
df = convert_data_to_dataframe(iris)


# ---- Gráficos -----
left_column, right_column = st.columns(2)
with left_column:
    with st.container():
        plot_scatter_graph(
            df, df_input, x="Comprimento da pétala", y="Largura da pétala")


with right_column:
    with st.container():
        plot_scatter_graph(
            df, df_input, x="Comprimento da sépala", y="Largura da sépala")

st.markdown("---")
