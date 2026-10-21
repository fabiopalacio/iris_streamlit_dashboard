import pandas as pd
import streamlit as st


def create_expander_section():
    with st.expander("Sobre a aplicação"):
        st.markdown("<p>Aplicação que utiliza um modelo de <b>RandomForest</b> para prever a qual espécie do gênero de <i>Iris</i> as medidas inseridas pelo usuário pertence.</p>"
                    "<p>As medidas obtidas pelo usuário devem ser inseridas no menu lateral.", True)

    st.markdown("---")


def create_sidebar():

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
    return df_input
