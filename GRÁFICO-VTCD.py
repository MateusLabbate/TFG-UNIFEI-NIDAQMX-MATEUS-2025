import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates 
import os

NOME_DO_ARQUIVO_CSV = "VALIDAÇÃO_20251016_164932_evento_2.csv" 

TENSAO_NOMINAL_V = 127.0
LIMITE_INFERIOR_PU = 0.9
LIMITE_SUPERIOR_PU = 1.1

def analisar_e_plotar_evento(nome_arquivo):
    """
    Carrega o CSV de evento VTCD e gera o gráfico.
    """
    print(f"Analisando o arquivo: {nome_arquivo}")

    limite_inferior_v = TENSAO_NOMINAL_V * LIMITE_INFERIOR_PU
    limite_superior_v = TENSAO_NOMINAL_V * LIMITE_SUPERIOR_PU

    try:
        df_evento = pd.read_csv(nome_arquivo)
        df_evento['Timestamp'] = pd.to_datetime(df_evento['Timestamp'])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df_evento['Timestamp'], df_evento['V_True_RMS'], color='black', linewidth=1.5, label='Tensão RMS Medida')
        ax.axhline(y=limite_inferior_v, color='red', linestyle='--', linewidth=1.5, label=f'Limiar Afundamento 0.9 PU ({limite_inferior_v:.1f} V)')
        ax.axhline(y=limite_superior_v, color='blue', linestyle='--', linewidth=1.5, label=f'Limiar Elevação 1.1 PU ({limite_superior_v:.1f} V)')

        ax.set_title('Análise de Variação de Tensão de Curta Duração (VTCD)', fontsize=16, pad=15)
        ax.set_xlabel('Tempo', fontsize=12)
        ax.set_ylabel('Tensão Eficaz (Vrms)', fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f V'))

        date_formatter = mdates.DateFormatter('%H:%M:%S.%f')

        ax.xaxis.set_major_formatter(date_formatter)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        ax.legend(loc='lower right')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        max_voltage = df_evento['V_True_RMS'].max()
        top_limit = max(limite_superior_v * 1.05, max_voltage * 1.05)
        ax.set_ylim(bottom=0, top=top_limit)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    analisar_e_plotar_evento(NOME_DO_ARQUIVO_CSV)