
def plot_scatter_graph(ax, df, x, y):

    for species in df["Espécie"].unique():
        dados = df[df["Espécie"] == species]

        ax.scatter(
            dados[x],
            dados[y],
            label=species,
            alpha=0.7
        )

    ax.set_xlabel(f"{x}")
    ax.set_ylabel(f"{y}")
    ax.set_title("Medidas de flores do gênero Iris")
    ax.legend()
    ax.patch.set_facecolor('#f1fff7')
    ax.grid(alpha=0.2)
