import streamlit as st
from sklearn import datasets
from src.layout import create_expander_section, create_sidebar
from src.tools import convert_data_to_dataframe, make_prediction, plot_scatter_graph, set_title, write_user_data_as_table

st.set_page_config(
    page_title="Predição Iris",
    layout="wide"
)

iris = datasets.load_iris()


# --- Instruções da aplicação como expander ----
create_expander_section()

# ---- Barra lateral com inputs dos dados do usuário -----
df_input = create_sidebar()


# ---- Aplicação do classificador aos dados inseridos ----
prediction, prediction_proba = make_prediction(df_input)
species_predicted = "Iris " + (
    str(iris.target_names[prediction])
    .replace("'", "")
    .replace("[", "")
    .replace("]", "")
)

# ---- Criação das colunas com os dados inseridos (esquerda) e resultado do algoritmo (direita) -----
left_column, right_column = st.columns(2)
with left_column:
    with st.container():
        st.markdown(
            set_title("h3", "center", "Parâmetros inseridos"), True)
        st.table(write_user_data_as_table(df_input))

with right_column:
    with st.container():
        st.markdown(
            set_title("h3", "center", "Espécie encontrada:"), True)

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
