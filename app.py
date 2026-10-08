
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

df = pd.read_csv('ecommerce_estatistica.csv')


def criar_histograma(df):
    fig = px.histogram(
        df,
        x='Nota',
        title='Distribuição das Notas'
    )

    fig.update_layout(
        xaxis_title='Nota',
        yaxis_title='Quantidade de Produtos'
    )

    return fig


def criar_dispersao(df):
    fig = px.scatter(
        df,
        x='N_Avaliações',
        y='Preço',
        title='Relação entre Avaliações e Preço'
    )

    fig.update_layout(
        xaxis_title='Número de avaliações',
        yaxis_title='Preço do produto'
    )

    return fig

def criar_heatmap(df):
    # Selecionar as colunas numéricas
    colunas = df[['Nota', 'N_Avaliações', 'Desconto', 'Preço']]

    # Calcular a correlação
    correlacao = colunas.corr()

    # Criar o mapa de calor
    fig = px.imshow(
        correlacao,
        text_auto=True,
        title='Mapa de Calor das Correlações'
    )

    return fig

def criar_barras(df):
    # Contar quantos produtos existem por marca
    marcas = df['Marca'].value_counts().head(10)

    # Criar o gráfico de barras
    fig = px.bar(
        x=marcas.index,
        y=marcas.values,
        title='Top 10 Marcas por Quantidade de Produtos'
    )

    fig.update_layout(
        xaxis_title='Marca',
        yaxis_title='Quantidade de Produtos'
    )

    return fig

def criar_pizza(df):
    # Contar os gêneros e selecionar os 5 mais frequentes
    generos = df['Gênero'].value_counts().head(5)

    # Criar o gráfico de pizza
    fig = px.pie(
        names=generos.index,
        values=generos.values,
        title='Distribuição dos 5 Principais Gêneros'
    )

    return fig


def criar_densidade(df):
    # Selecionar os preços e remover valores ausentes
    dados = df['Preço'].dropna()

    # Calcular a densidade
    kde = gaussian_kde(dados)

    # Criar os pontos do eixo X
    x = np.linspace(dados.min(), dados.max(), 200)

    # Calcular a densidade para cada ponto
    y = kde(x)

    # Criar o gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        fill='tozeroy',
        name='Densidade'
    ))

    fig.update_layout(
        title='Distribuição dos Preços',
        xaxis_title='Preço (R$)',
        yaxis_title='Densidade'
    )

    return fig


def criar_regressao(df):
    fig = px.scatter(
        df,
        x='N_Avaliações',
        y='Preço',
        trendline='ols',
        title='Regressão: Avaliações x Preço'
    )

    fig.update_layout(
        xaxis_title='Número de avaliações',
        yaxis_title='Preço do produto (R$)'
    )

    return fig

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Dashboard de E-commerce'),

    dcc.Graph(figure=criar_histograma(df)),
    dcc.Graph(figure=criar_dispersao(df)),
    dcc.Graph(figure=criar_heatmap(df)),
    dcc.Graph(figure=criar_barras(df)),
    dcc.Graph(figure=criar_pizza(df)),
    dcc.Graph(figure=criar_densidade(df)),
    dcc.Graph(figure=criar_regressao(df))
])


if __name__ == '__main__':
    app.run(debug=True)
