import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

#1. PARÂMETROS INICIAIS

NOME_ARQUIVO_ENTRADA = 'MEDIÇÃO_OFICIAL_20251007_162600.csv'
nome_base, extensao = os.path.splitext(NOME_ARQUIVO_ENTRADA)
NOME_ARQUIVO_SAIDA = f"{nome_base}_analise_QEE.csv"
TENSAO_NOMINAL_KV = 0.127
N_BASE_PRODIST = 1008

LIMITES_TENSAO = {
    'critica_inferior': 110.0,
    'precaria_inferior': 117.0,
    'precaria_superior': 133.0,
    'critica_superior': 135.0
}

LIMITE_DRP = 3.0
LIMITE_DRC = 0.5

ORDENS_HARMONICAS_TOTAIS = list(range(2, 41))
HARMONICAS_DESTAQUE = {
    3: {'color': 'red', 'style': '-', 'label': 'DIT 3ª Ordem (%)'},
    5: {'color': 'green', 'style': '-', 'label': 'DIT 5ª Ordem (%)'},
    7: {'color': 'purple', 'style': '-', 'label': 'DIT 7ª Ordem (%)'}
}

GERAR_GRAFICO_TENSAO = True
GERAR_GRAFICO_HARMONICOS = True
GERAR_GRAFICO_DISTORCAO = True 

#2. FUNÇÕES DE PLOTAGEM
def plotar_serie_temporal_tensao(df_dados, limites_tensao):
    """
    Gera um gráfico da série temporal da Tensão RMS Total (True RMS).
    """
    print("\n" + "="*60)
    print("--- GERANDO GRÁFICO DA SÉRIE TEMPORAL DE TENSÃO ---")
    try:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(df_dados['Timestamp_Fim'], df_dados['Tensao_RMS_Total_V'], 
                label='Tensão RMS Total (True RMS)', 
                color='#007ACC', 
                linewidth=1.5)
        ax.axhline(y=limites_tensao['precaria_superior'], color='gold', linestyle='--', label='Limites Precários')
        ax.axhline(y=limites_tensao['precaria_inferior'], color='gold', linestyle='--')
        ax.axhline(y=limites_tensao['critica_superior'], color='red', linestyle='--', label='Limites Críticos')
        ax.axhline(y=limites_tensao['critica_inferior'], color='red', linestyle='--')
        ax.set_title('Série Temporal da Tensão RMS Total', fontsize=18, pad=20)
        ax.set_xlabel('Data e Hora', fontsize=12)
        ax.set_ylabel('Tensão (Vrms)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.margins(x=0.01)
        fig.tight_layout()

        nome_grafico = 'grafico_serie_temporal_tensao_RMS.png'
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo com sucesso como '{nome_grafico}'")
    except Exception as e:
        print(f"ERRO ao gerar o gráfico de tensão: {e}")

def plotar_espectro_harmonico(df_dados):
    """
    Gera um gráfico de barras do espectro harmônico (Percentil 95 de cada DIT).
    """
    print("\n" + "="*60)
    print("--- GERANDO GRÁFICO DO ESPECTRO HARMÔNICO ---")
    try:
        ordens = []
        valores_p95 = []
        for h in ORDENS_HARMONICAS_TOTAIS:
            coluna_dit = f'DIT{h}_%'
            if coluna_dit in df_dados.columns:
                ordens.append(h)
                p95 = np.percentile(df_dados[coluna_dit].values, 95)
                valores_p95.append(p95)
        if not ordens:
            print("Nenhuma coluna de DIT encontrada para gerar o gráfico.")
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(ordens, valores_p95, color='#00529B', width=0.7, label='DIT (Percentil 95)')
        ax.set_title('Espectro Harmônico (Percentil 95)', fontsize=18, pad=20)
        ax.set_xlabel('Ordem Harmônica', fontsize=12)
        ax.set_ylabel('Distorção Harmônica Individual (DIT %)', fontsize=12)
        ax.grid(axis='y', linestyle=':', linewidth=0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xticks(ordens)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()

        nome_grafico = 'grafico_espectro_harmonico.png'
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo com sucesso como '{nome_grafico}'")
    except Exception as e:
        print(f"ERRO ao gerar o gráfico de espectro harmônico: {e}")

def plotar_serie_temporal_distorcao(df_dados):
    """
    Gera um gráfico da série temporal de Distorção Harmônica (DTT e DITs).
    """
    print("\n" + "="*60)
    print("--- GERANDO GRÁFICO DA SÉRIE TEMPORAL DE DISTORÇÃO ---")
    try:
        fig, ax = plt.subplots(figsize=(16, 8))

        if 'DTT_%' in df_dados.columns:
            ax.plot(df_dados['Timestamp_Fim'], df_dados['DTT_%'], 
                    label='Distorção Total (DTT%)', 
                    color='black', 
                    linewidth=2.5,
                    zorder=10)
        else:
            print("AVISO: Coluna 'DTT_%' não encontrada.")

        cores_fundo = plt.cm.viridis(np.linspace(0, 1, len(ORDENS_HARMONICAS_TOTAIS)))
        print("Plotando DITs de 2ª a 40ª ordem (com destaque em 3, 5, 7)...")
        
        for i, ordem in enumerate(ORDENS_HARMONICAS_TOTAIS):
            coluna_dit = f'DIT{ordem}_%'
            if coluna_dit not in df_dados.columns:
                continue 

            if ordem in HARMONICAS_DESTAQUE:
                info = HARMONICAS_DESTAQUE[ordem]
                ax.plot(df_dados['Timestamp_Fim'], df_dados[coluna_dit], 
                        label=info['label'],
                        color=info['color'], 
                        linestyle=info['style'],
                        linewidth=1.2,
                        zorder=9)
            else:
                ax.plot(df_dados['Timestamp_Fim'], df_dados[coluna_dit], 
                        color=cores_fundo[i], 
                        linewidth=0.5, 
                        alpha=0.5)

        ax.set_title('Série Temporal da Distorção Harmônica (DTT e DITs Individuais)', fontsize=18, pad=20)
        ax.set_xlabel('Data e Hora', fontsize=12)
        ax.set_ylabel('Distorção Harmônica (%)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle=':', linewidth=0.6)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        ax.margins(x=0.01)
        fig.tight_layout()
        
        nome_grafico = 'grafico_serie_temporal_distorcao_destaque.png'
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo com sucesso como '{nome_grafico}'")
    except Exception as e:
        print(f"ERRO ao gerar o gráfico de distorção temporal: {e}")

#3. NÚCLEO DO SCRIPT
print(f"--- Análise de Qualidade de Energia Elétrica (QEE) ---")

#CARREGAR OS DADOS
try:
    print(f"Lendo o arquivo de dados: '{NOME_ARQUIVO_ENTRADA}'...")
    df_harmonicos = pd.read_csv(NOME_ARQUIVO_ENTRADA)
    if df_harmonicos.empty:
        print("ERRO: O arquivo de dados está vazio.")
        exit()
    if len(df_harmonicos) < 20:
        print(f"AVISO: O arquivo de dados tem poucas linhas ({len(df_harmonicos)}). O cálculo do percentil 95 pode não ser estatisticamente relevante.")
    print("Arquivo carregado com sucesso.")
except FileNotFoundError:
    print(f"ERRO: O arquivo '{NOME_ARQUIVO_ENTRADA}' não foi encontrado.")
    exit()

print("Iniciando os cálculos...")
v_fundamental = df_harmonicos['H1_Vrms']
colunas_identificacao = ['Intervalo_N', 'Timestamp_Fim', 'Duracao_Min']
df_resultados = df_harmonicos[colunas_identificacao].copy()
df_resultados['Timestamp_Fim'] = pd.to_datetime(df_resultados['Timestamp_Fim'])

#CÁLCULOS QEE
colunas_v_harmonicos = [f'H{h}_Vrms' for h in ORDENS_HARMONICAS_TOTAIS if f'H{h}_Vrms' in df_harmonicos.columns]
soma_dos_quadrados_harmonicos = pd.Series(0, index=df_harmonicos.index)
if colunas_v_harmonicos:
    soma_dos_quadrados_harmonicos = (df_harmonicos[colunas_v_harmonicos] ** 2).sum(axis=1)

print("--> Calculando Tensão RMS Total (V_rms)...")
soma_total_quadrado = v_fundamental**2 + soma_dos_quadrados_harmonicos
tensao_rms_total = np.sqrt(soma_total_quadrado)
df_resultados['Tensao_RMS_Total_V'] = tensao_rms_total

print("--> Classificando faixas de tensão...")
condicoes = [
    (df_resultados['Tensao_RMS_Total_V'] < LIMITES_TENSAO['critica_inferior']) | (df_resultados['Tensao_RMS_Total_V'] >= LIMITES_TENSAO['critica_superior']),
    (df_resultados['Tensao_RMS_Total_V'] >= LIMITES_TENSAO['precaria_inferior']) & (df_resultados['Tensao_RMS_Total_V'] < LIMITES_TENSAO['precaria_superior']),
    (df_resultados['Tensao_RMS_Total_V'] >= LIMITES_TENSAO['critica_inferior']) & (df_resultados['Tensao_RMS_Total_V'] < LIMITES_TENSAO['precaria_inferior']),
    (df_resultados['Tensao_RMS_Total_V'] >= LIMITES_TENSAO['precaria_superior']) & (df_resultados['Tensao_RMS_Total_V'] < LIMITES_TENSAO['critica_superior'])
]
opcoes = ['Crítica', 'Adequada', 'Precária', 'Precária']
df_resultados['Faixa_Tensao'] = np.select(condicoes, opcoes, default='Indefinida')

print("--> Calculando DTT%...")
dtt_percentual = np.divide(np.sqrt(soma_dos_quadrados_harmonicos), v_fundamental, out=np.zeros_like(v_fundamental), where=v_fundamental!=0) * 100
df_resultados['DTT_%'] = dtt_percentual

print("--> Calculando DTTp%...")
ordens_dttp = [h for h in ORDENS_HARMONICAS_TOTAIS if h % 2 == 0 and h % 3 != 0]
colunas_v_harmonicos_dttp = [f'H{h}_Vrms' for h in ordens_dttp if f'H{h}_Vrms' in df_harmonicos.columns]
if colunas_v_harmonicos_dttp:
    soma_dos_quadrados_dttp = (df_harmonicos[colunas_v_harmonicos_dttp] ** 2).sum(axis=1)
    dttp_percentual = np.divide(np.sqrt(soma_dos_quadrados_dttp), v_fundamental, out=np.zeros_like(v_fundamental), where=v_fundamental!=0) * 100
    df_resultados['DTTp_%'] = dttp_percentual

print("--> Calculando DTTi%...")
ordens_dtti = [h for h in ORDENS_HARMONICAS_TOTAIS if h % 2 != 0 and h % 3 != 0]
colunas_v_harmonicos_dtti = [f'H{h}_Vrms' for h in ordens_dtti if f'H{h}_Vrms' in df_harmonicos.columns]
if colunas_v_harmonicos_dtti:
    soma_dos_quadrados_dtti = (df_harmonicos[colunas_v_harmonicos_dtti] ** 2).sum(axis=1)
    dtti_percentual = np.divide(np.sqrt(soma_dos_quadrados_dtti), v_fundamental, out=np.zeros_like(v_fundamental), where=v_fundamental!=0) * 100
    df_resultados['DTTi_%'] = dtti_percentual

print("--> Calculando DTT3%...")
ordens_dtt3 = [h for h in ORDENS_HARMONICAS_TOTAIS if h % 3 == 0]
colunas_v_harmonicos_dtt3 = [f'H{h}_Vrms' for h in ordens_dtt3 if f'H{h}_Vrms' in df_harmonicos.columns]
if colunas_v_harmonicos_dtt3:
    soma_dos_quadrados_dtt3 = (df_harmonicos[colunas_v_harmonicos_dtt3] ** 2).sum(axis=1)
    dtt3_percentual = np.divide(np.sqrt(soma_dos_quadrados_dtt3), v_fundamental, out=np.zeros_like(v_fundamental), where=v_fundamental!=0) * 100
    df_resultados['DTT3_%'] = dtt3_percentual

print("--> Calculando DIT%...")
for ordem in ORDENS_HARMONICAS_TOTAIS:
    coluna_harmonico_vrms = f'H{ordem}_Vrms'
    coluna_saida_dit = f'DIT{ordem}_%'
    if coluna_harmonico_vrms in df_harmonicos.columns:
        v_harmonico = df_harmonicos[coluna_harmonico_vrms]
        dit_percentual = np.divide(v_harmonico, v_fundamental, out=np.zeros_like(v_harmonico), where=v_fundamental!=0) * 100
        df_resultados[coluna_saida_dit] = dit_percentual

#4. RELATÓRIOS
print("\n" + "="*60)
print("--- ANÁLISE DE VARIAÇÃO DE TENSÃO (REGIME PERMANENTE) ---")
contagem_faixas = df_resultados['Faixa_Tensao'].value_counts()
NLT = len(df_resultados)
nlp = contagem_faixas.get('Precária', 0)
nlc = contagem_faixas.get('Crítica', 0)
drp_percentual = (nlp / N_BASE_PRODIST) * 100
drc_percentual = (nlc / N_BASE_PRODIST) * 100
status_drp = "DENTRO do limite" if drp_percentual <= LIMITE_DRP else ">> FORA DO LIMITE <<"
status_drc = "DENTRO do limite" if drc_percentual <= LIMITE_DRC else ">> FORA DO LIMITE <<"
print(f"Número de Leituras no Arquivo: {NLT}")
print(f"Base de Cálculo (PRODIST): {N_BASE_PRODIST} leituras (1 semana)")
print(f"Leituras em Faixa Precária (nlp): {nlp}")
print(f"Leituras em Faixa Crítica (nlc): {nlc}")
print("-"*60)
print("Indicadores de Conformidade DRP e DRC:")
print(f"Indicador: DRP  | Calculado: {drp_percentual:5.2f}% | Limite PRODIST: {LIMITE_DRP:4.1f}% | Status: {status_drp}")
print(f"Indicador: DRC  | Calculado: {drc_percentual:5.2f}% | Limite PRODIST: {LIMITE_DRC:4.1f}% | Status: {status_drc}")


print("\n" + "="*60)
print("--- ANÁLISE DE CONFORMIDADE HARMÔNICA (PERCENTIL 95) ---")
print(f"Tensão Nominal para Análise: {TENSAO_NOMINAL_KV} kV")
LIMITES_PRODIST = {
    'ate_1kV':     {'DTT': 10.0, 'DTTp': 2.5, 'DTTi': 7.5, 'DTT3': 6.5},
    '1_a_69kV':    {'DTT': 8.0,  'DTTp': 2.0, 'DTTi': 6.0, 'DTT3': 5.0},
    '69_a_230kV':  {'DTT': 5.0,  'DTTp': 1.0, 'DTTi': 4.0, 'DTT3': 3.0}
}
if TENSAO_NOMINAL_KV <= 1.0:
    limites_selecionados = LIMITES_PRODIST['ate_1kV']
    faixa_tensao = "Vn <= 1,0 kV"
elif 1.0 < TENSAO_NOMINAL_KV < 69.0:
    limites_selecionados = LIMITES_PRODIST['1_a_69kV']
    faixa_tensao = "1,0 kV < Vn < 69 kV"
elif 69.0 <= TENSAO_NOMINAL_KV <= 230.0:
    limites_selecionados = LIMITES_PRODIST['69_a_230kV']
    faixa_tensao = "69 kV <= Vn <= 230 kV"
else:
    limites_selecionados = None
    faixa_tensao = "Não definida"
if limites_selecionados:
    print(f"Faixa de Tensão PRODIST Aplicada: {faixa_tensao}")
    print("-"*60)
    print("Análise de Distorções Harmônicas:")
    indicadores_para_analise = {'DTT': 'DTT_%', 'DTTp': 'DTTp_%', 'DTTi': 'DTTi_%', 'DTT3': 'DTT3_%'}
    for nome_indicador, coluna in indicadores_para_analise.items():
        if coluna in df_resultados:
            p95_calculado = np.percentile(df_resultados[coluna].values, 95)
            limite = limites_selecionados[nome_indicador]
            if p95_calculado <= limite:
                status = "DENTRO do limite"
            else:
                status = ">> FORA DO LIMITE <<"
            print(f"Indicador: {nome_indicador: <5} | P95 Calculado: {p95_calculado:5.2f}% | Limite PRODIST: {limite:4.1f}% | Status: {status}")
        else:
            print(f"Indicador: {nome_indicador: <5} | Coluna '{coluna}' não encontrada para análise.")
else:
    print(f"AVISO: Tensão nominal de {TENSAO_NOMINAL_KV} kV está fora das faixas definidas pelo PRODIST. Nenhuma análise de conformidade será realizada.")
print("="*60 + "\n")

#5. SALVANDO OS RESULTADOS NO ARQUIVO DE SAÍDA
try:
    print("Salvando arquivo CSV com todos os cálculos...")
    df_resultados.to_csv(NOME_ARQUIVO_SAIDA, index=False, float_format='%.4f')
    print("-" * 60)
    print(f"Análise finalizada com sucesso!")
    print(f"Arquivo de resultados detalhados salvo em: '{NOME_ARQUIVO_SAIDA}'")
    print("-" * 60)
except Exception as e:
    print(f"ERRO: Não foi possível salvar o arquivo de resultados. Motivo: {e}")

if GERAR_GRAFICO_TENSAO:
    plotar_serie_temporal_tensao(df_resultados, LIMITES_TENSAO)

if GERAR_GRAFICO_HARMONICOS:
    plotar_espectro_harmonico(df_resultados)

if GERAR_GRAFICO_DISTORCAO:
    plotar_serie_temporal_distorcao(df_resultados)

if GERAR_GRAFICO_TENSAO or GERAR_GRAFICO_HARMONICOS or GERAR_GRAFICO_DISTORCAO:
    plt.show()