
import yfinance as yf
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

def get_data(tickers):
    df = pd.DataFrame()
    for ticker in tickers:
        symbol = ticker if ticker.endswith('.SA') or ticker == '^BVSP' else ticker + '.SA'
        data = yf.download(symbol, start='2022-01-01')['Close']
        df[ticker] = data
    df.dropna(inplace=True)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.json
    tickers = data.get('tickers', [])
    metodo = data.get('metodo', 'retorno')

    df = get_data(tickers)
    output = {}
    graph_json = ""

    if metodo == 'retorno':
        retorno = ((df.iloc[-1] / df.iloc[0]) - 1) * 100
        output = retorno.round(2).to_dict()
        fig = go.Figure()
        for ticker in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ticker], name=ticker))
        fig.update_layout(title="Histórico de Preços", template="plotly_dark")
        graph_json = pio.to_json(fig)

    elif metodo == 'risco':
        retorno = df.pct_change().dropna()
        risco = retorno.std() * np.sqrt(252) * 100
        output = risco.round(2).to_dict()
        fig = go.Figure(data=[go.Bar(x=risco.index, y=risco.values)])
        fig.update_layout(title="Volatilidade Anualizada (%)", template="plotly_dark")
        graph_json = pio.to_json(fig)

    elif metodo == 'comparativo':
        if 'IBOV' not in df.columns:
            df['IBOV'] = get_data(['^BVSP'])['^BVSP']
        variacao = ((df[tickers[0]] - df[tickers[0]].shift(1)) / df[tickers[0]].shift(1)) * 100
        variacao_ibov = ((df['IBOV'] - df['IBOV'].shift(1)) / df['IBOV'].shift(1)) * 100
        diff = variacao - variacao_ibov
        df['Diferença %'] = diff
        output = {"diferença média": round(diff.mean(), 2)}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Diferença %'], name="Diferença Ativo x IBOV"))
        fig.update_layout(title="Diferença percentual diária vs IBOV", template="plotly_dark")
        graph_json = pio.to_json(fig)

    elif metodo == 'capm':
        df['IBOV'] = get_data(['^BVSP'])['^BVSP']
        df.dropna(inplace=True)
        retorno = np.log(df / df.shift(1)).dropna()
        betas = {}
        rf = 0.1375  # taxa selic exemplo
        rm = retorno['IBOV'].mean() * 252
        for ticker in tickers:
            if ticker == 'IBOV': continue
            cov = np.cov(retorno[ticker], retorno['IBOV'])[0][1]
            var = np.var(retorno['IBOV'])
            beta = cov / var
            capm = rf + beta * (rm - rf)
            betas[ticker] = round(capm * 100, 2)
        output = betas
        fig = go.Figure(data=[go.Bar(x=list(betas.keys()), y=list(betas.values()))])
        fig.update_layout(title="Retorno Esperado pelo CAPM (%)", template="plotly_dark")
        graph_json = pio.to_json(fig)

    elif metodo == 'otimizacao':
        df.dropna(inplace=True)
        retornos = np.log(df / df.shift(1)).dropna()
        n = len(df.columns)
        pesos = np.random.random(n)
        pesos /= pesos.sum()
        cov = retornos.cov() * 252
        retorno_esperado = np.sum(retornos.mean() * pesos) * 252
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        sharpe = (retorno_esperado - 0.1375) / volatilidade
        output = {
            "Retorno Esperado (%)": round(retorno_esperado * 100, 2),
            "Volatilidade (%)": round(volatilidade * 100, 2),
            "Sharpe Ratio": round(sharpe, 2)
        }
        fig = go.Figure(data=[go.Pie(labels=df.columns, values=pesos, hole=.3)])
        fig.update_layout(title="Alocação Ótima Simulada", template="plotly_dark")
        graph_json = pio.to_json(fig)

    return jsonify({'dados': output, 'grafico': graph_json})

if __name__ == '__main__':
    app.run(debug=True)
