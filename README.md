
# ARGA Quant - Web App

Este é um aplicativo web interativo para análise de ações brasileiras, incluindo:

- Cálculo de Retorno
- Análise de Risco (Volatilidade)
- Comparação com IBOVESPA
- Precificação por CAPM
- Otimização de Portfólio (com Sharpe Ratio)

## Como rodar no Replit

1. Crie um novo repositório no Replit.
2. Faça upload de todos os arquivos deste projeto.
3. Certifique-se de que `.replit` contém o comando:
   ```
   run = "gunicorn app:app --bind=0.0.0.0:8000"
   ```
4. Clique em **Run** para iniciar o servidor.

## Requisitos

- Python 3.11
- Flask
- yfinance
- pandas
- numpy
- plotly
- gunicorn
