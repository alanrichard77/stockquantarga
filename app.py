
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

def baixar_dados(tickers, inicio, fim):
    df = yf.download(tickers, start=inicio, end=fim)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna()

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    grafico_json = None
    if request.method == 'POST':
        tickers = request.form['tickers'].upper().replace(" ", "").split(',')
        inicio = request.form['inicio']
        fim = request.form['fim']
        tipo = request.form['tipo']
        df = baixar_dados(tickers + ['^BVSP'], inicio, fim)
        df.columns = [t.replace('.SA', '') if '.SA' in t else t for t in df.columns]
        df.dropna(inplace=True)

        if tipo == 'retorno':
            retorno_simples = ((df.iloc[-1] / df.iloc[0]) - 1) * 100
            retorno_log = np.log(df / df.shift(1)).mean() * 252 * 100
            resultado = {
                "Retorno Simples (%)": retorno_simples.round(2).to_dict(),
                "Retorno Log (%)": retorno_log.round(2).to_dict()
            }
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
            fig.update_layout(title="Histórico de Preços", template="plotly_dark")
            grafico_json = pio.to_html(fig, full_html=False)

        elif tipo == 'risco':
            retorno = df.pct_change().dropna()
            vol_anual = retorno.std() * np.sqrt(252) * 100
            cov = retorno.cov()
            corr = retorno.corr()
            resultado = {
                "Volatilidade (%)": vol_anual.round(2).to_dict(),
                "Covariância": cov.to_dict(),
                "Correlação": corr.to_dict()
            }
            fig = go.Figure(data=[go.Bar(x=vol_anual.index, y=vol_anual.values)])
            fig.update_layout(title="Volatilidade Anualizada (%)", template="plotly_dark")
            grafico_json = pio.to_html(fig, full_html=False)

        elif tipo == 'comparativo':
            ativo = tickers[0]
            df['DIF'] = df[ativo].pct_change() - df['^BVSP'].pct_change()
            resultado = {"Média Diferença %": round(df['DIF'].mean() * 100, 2)}
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['DIF'] * 100, name='Diferença %'))
            fig.update_layout(title=f'Diferença Diária %: {ativo} x IBOV', template="plotly_dark")
            grafico_json = pio.to_html(fig, full_html=False)

        elif tipo == 'capm':
            retorno = np.log(df / df.shift(1)).dropna()
            rf = 0.1375  # taxa livre de risco (13.75%)
            rm = retorno['^BVSP'].mean() * 252
            capms = {}
            for ticker in tickers:
                if ticker == '^BVSP': continue
                beta = retorno[ticker].cov(retorno['^BVSP']) / retorno['^BVSP'].var()
                capm = rf + beta * (rm - rf)
                capms[ticker] = round(capm * 100, 2)
            resultado = {"CAPM (%)": capms}
            fig = go.Figure(data=[go.Bar(x=list(capms.keys()), y=list(capms.values()))])
            fig.update_layout(title="Retorno Esperado pelo CAPM (%)", template="plotly_dark")
            grafico_json = pio.to_html(fig, full_html=False)

        elif tipo == 'otimizacao':
            retorno = np.log(df / df.shift(1)).dropna()
            num_assets = len(tickers)
            pesos = np.random.random(num_assets)
            pesos /= pesos.sum()
            mean_returns = retorno[tickers].mean() * 252
            cov_matrix = retorno[tickers].cov() * 252
            retorno_esperado = np.dot(pesos, mean_returns)
            volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
            sharpe = (retorno_esperado - 0.1375) / volatilidade
            resultado = {
                "Retorno Esperado (%)": round(retorno_esperado * 100, 2),
                "Volatilidade (%)": round(volatilidade * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Pesos (%)": dict(zip(tickers, (pesos * 100).round(2)))
            }
            fig = go.Figure(data=[go.Pie(labels=tickers, values=pesos, hole=.3)])
            fig.update_layout(title="Alocação Otimizada (simulada)", template="plotly_dark")
            grafico_json = pio.to_html(fig, full_html=False)

    return render_template("index.html", resultado=resultado, grafico=grafico_json)

if __name__ == '__main__':
    app.run(debug=True)
