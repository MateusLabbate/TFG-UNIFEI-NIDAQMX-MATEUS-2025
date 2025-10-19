import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import datetime
import csv
import numpy as np
import pandas as pd
import os
import collections

#1. PARÂMETROS
CANAL = "Dev2/ai0"
TAXA_AMOSTRAGEM = 24000
AMOSTRAS_POR_LEITURA_DAQ = 4800
CONSTANTE_TRANSDUTOR = 99

FREQUENCIA_FUNDAMENTAL_HZ = 60
NUM_CICLOS_JANELA_FFT = 12
INTERVALO_AGREGACAO_MIN = 10

TENSAO_NOMINAL_V = 127.0
LIMITE_INFERIOR_PU = 0.9
LIMITE_SUPERIOR_PU = 1.1
PRE_GATILHO_CICLOS = 30
POS_GATILHO_CICLOS = 30

DURACAO_SEGUNDOS = 604800

base_arquivo_saida = f"MEDIÇÃO_OFICIAL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
arquivo_saida_harmonicos = f"{base_arquivo_saida}.csv"

ORDENS_HARMONICAS_ANALISAR = list(range(1, 41))
AMOSTRAS_POR_CICLO = int(TAXA_AMOSTRAGEM / FREQUENCIA_FUNDAMENTAL_HZ)
AMOSTRAS_POR_JANELA_FFT = int(NUM_CICLOS_JANELA_FFT * AMOSTRAS_POR_CICLO)
DURACAO_JANELA_FFT_S = NUM_CICLOS_JANELA_FFT / FREQUENCIA_FUNDAMENTAL_HZ
JANELAS_POR_INTERVALO_10MIN = int((INTERVALO_AGREGACAO_MIN * 60) / DURACAO_JANELA_FFT_S)
LIMITE_INFERIOR_V = TENSAO_NOMINAL_V * LIMITE_INFERIOR_PU
LIMITE_SUPERIOR_V = TENSAO_NOMINAL_V * LIMITE_SUPERIOR_PU

buffer_montagem_ciclos = np.array([])
buffer_janela_12_ciclos = []
resultados_janelas_temp = []
amostras_processadas_total = 0

evento_vtcd_contador = 0
em_evento_vtcd = False
pos_gatilho_contador = 0
buffer_circular_ciclos = collections.deque(maxlen=PRE_GATILHO_CICLOS)
dados_evento_atual = []

intervalo_harmonico_atual = 1
intervalo_harmonico_contaminado = False
duracao_total_ajustada_s = DURACAO_SEGUNDOS

colunas_harmonicas = [f'H{ordem}_Vrms' for ordem in ORDENS_HARMONICAS_ANALISAR]
cabecalho_harmonicos_csv = ['Intervalo_N', 'Timestamp_Fim', 'Duracao_Min'] + colunas_harmonicas
with open(arquivo_saida_harmonicos, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cabecalho_harmonicos_csv)

print("--- Medidor de Qualidade de Energia ---")
print(f"Duração base da medição: {DURACAO_SEGUNDOS / 3600:.1f} horas")
print(f"Limites VTCD: Afundamento < {LIMITE_INFERIOR_V:.1f} V | Elevação > {LIMITE_SUPERIOR_V:.1f} V")
print(f"Resultados de Harmônicas serão salvos em: {arquivo_saida_harmonicos}")
print("-" * 40)

#2. FUNÇÕES AUXILIARES
def calcular_true_rms(ciclo_dados):
    fft_resultado = np.fft.fft(ciclo_dados)
    N = len(ciclo_dados)
    N_metade = N // 2
    magnitudes_pico = (2.0 / N) * np.abs(fft_resultado[0:N_metade])
    tensoes_rms_h = magnitudes_pico / np.sqrt(2)
    soma_quadrados = np.sum(tensoes_rms_h[1:len(ORDENS_HARMONICAS_ANALISAR)+1]**2)
    return np.sqrt(soma_quadrados)

def salvar_evento_csv(dados_do_evento, num_evento, timestamp_base):
    nome_arquivo = f"{timestamp_base}_evento_{num_evento}.csv"
    print(f"  -> Salvando valores True RMS do evento em: {nome_arquivo}")
    cabecalho = ['Timestamp', 'V_True_RMS']
    with open(nome_arquivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
        for info_ciclo in dados_do_evento:
            writer.writerow([info_ciclo['timestamp'].isoformat(), f"{info_ciclo['v_true_rms']:.4f}"])

#4. LOOP PRINCIPAL
try:
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(CANAL, min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(rate=TAXA_AMOSTRAGEM, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=AMOSTRAS_POR_LEITURA_DAQ)
        task.start()
        print("Aquisição iniciada... Pressione Ctrl+C para parar.")
        tempo_inicial_script = datetime.datetime.now()
        tempo_loop_inicial = time.time()

        while (time.time() - tempo_loop_inicial) < duracao_total_ajustada_s:
            dados_brutos = task.read(number_of_samples_per_channel=AMOSTRAS_POR_LEITURA_DAQ)
            dados_reais = np.array(dados_brutos) * CONSTANTE_TRANSDUTOR
            buffer_montagem_ciclos = np.concatenate((buffer_montagem_ciclos, dados_reais))
            
            while len(buffer_montagem_ciclos) >= AMOSTRAS_POR_CICLO:
                ciclo_atual_dados = buffer_montagem_ciclos[:AMOSTRAS_POR_CICLO]
                buffer_montagem_ciclos = buffer_montagem_ciclos[AMOSTRAS_POR_CICLO:]
                
                v_true_rms_ciclo = calcular_true_rms(ciclo_atual_dados)
                timestamp_ciclo = tempo_inicial_script + datetime.timedelta(seconds=(amostras_processadas_total / TAXA_AMOSTRAGEM))
                info_ciclo_atual = {'timestamp': timestamp_ciclo, 'v_true_rms': v_true_rms_ciclo}
                amostras_processadas_total += AMOSTRAS_POR_CICLO

                # LÓGICA DE EVENTOS DE VTCD
                if not em_evento_vtcd:
                    buffer_circular_ciclos.append(info_ciclo_atual)
                    if v_true_rms_ciclo < LIMITE_INFERIOR_V or v_true_rms_ciclo > LIMITE_SUPERIOR_V:
                        em_evento_vtcd = True
                        evento_vtcd_contador += 1
                        tipo_evento = "Afundamento" if v_true_rms_ciclo < LIMITE_INFERIOR_V else "Elevação"
                        print(f"\n(!) Evento de VTCD #{evento_vtcd_contador} detectado: {tipo_evento}")
                        
                        #EXPURGO E COMPENSAÇÃO
                        if not intervalo_harmonico_contaminado:
                            intervalo_harmonico_contaminado = True
                            duracao_total_ajustada_s += (INTERVALO_AGREGACAO_MIN * 60)
                            print(f"  -> Intervalo de harmônicas contaminado. Duração total estendida para {duracao_total_ajustada_s / 3600:.2f} horas.")

                        dados_evento_atual = list(buffer_circular_ciclos)
                else:
                    dados_evento_atual.append(info_ciclo_atual)
                    if LIMITE_INFERIOR_V < v_true_rms_ciclo < LIMITE_SUPERIOR_V:
                        if pos_gatilho_contador == 0:
                           pos_gatilho_contador = POS_GATILHO_CICLOS 
                    if pos_gatilho_contador > 0:
                        pos_gatilho_contador -= 1
                        if pos_gatilho_contador == 0:
                            salvar_evento_csv(dados_evento_atual, evento_vtcd_contador, base_arquivo_saida)
                            dados_evento_atual.clear()
                            em_evento_vtcd = False

                #ANÁLISES DE HARMÔNICOS
                buffer_janela_12_ciclos.append(ciclo_atual_dados)
                if len(buffer_janela_12_ciclos) == NUM_CICLOS_JANELA_FFT:
                    janela_completa = np.concatenate(buffer_janela_12_ciclos)
                    
                    fft_resultado = np.fft.fft(janela_completa)
                    freqs = np.fft.fftfreq(AMOSTRAS_POR_JANELA_FFT, 1 / TAXA_AMOSTRAGEM)
                    N_metade = AMOSTRAS_POR_JANELA_FFT // 2
                    magnitudes_pico = (2.0 / AMOSTRAS_POR_JANELA_FFT) * np.abs(fft_resultado[0:N_metade])

                    resultados_12c = {}
                    for ordem in ORDENS_HARMONICAS_ANALISAR:
                        idx = np.argmin(np.abs(freqs - (ordem * FREQUENCIA_FUNDAMENTAL_HZ)))
                        resultados_12c[f'H{ordem}_Vrms'] = magnitudes_pico[idx] / np.sqrt(2)
                    
                    resultados_janelas_temp.append(resultados_12c)
                    buffer_janela_12_ciclos.clear()

                    if len(resultados_janelas_temp) == JANELAS_POR_INTERVALO_10MIN:
                        #VERIFICAÇÃO DE INTERVALO CONTAMINADO
                        if intervalo_harmonico_contaminado:
                            print(f"-> {datetime.datetime.now().strftime('%H:%M:%S')}: Descartando intervalo de harmônicas #{intervalo_harmonico_atual} (contaminado por VTCD).")
                        else:
                            print(f"-> {datetime.datetime.now().strftime('%H:%M:%S')}: Agregando intervalo de harmônicas #{intervalo_harmonico_atual}...")
                            df_temp = pd.DataFrame(resultados_janelas_temp)
                            valores_agregados = np.sqrt((df_temp[colunas_harmonicas]**2).mean())
                            
                            resultado_final = valores_agregados.to_dict()
                            resultado_final['Intervalo_N'] = intervalo_harmonico_atual
                            resultado_final['Timestamp_Fim'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            resultado_final['Duracao_Min'] = INTERVALO_AGREGACAO_MIN
                            
                            with open(arquivo_saida_harmonicos, mode="a", newline="") as f:
                                writer = csv.DictWriter(f, fieldnames=cabecalho_harmonicos_csv)
                                writer.writerow(resultado_final)
                            intervalo_harmonico_atual += 1
                        
                        #RESETAR CONDIÇÃO
                        resultados_janelas_temp.clear()
                        intervalo_harmonico_contaminado = False

except KeyboardInterrupt:
    print("\nMedição interrompida pelo usuário.")
finally:
    print("\n--- Medição Finalizada ---")