import os
import sys
import requests
import time
import wave
import io
from pathlib import Path

# Chave API da AssemblyAI
API_KEY = "1b8d407761454280a91b55c5d0db1488"

def check_file(file_path):
    """
    Verifica se o arquivo existe e contém dados
    
    Args:
        file_path (str): Caminho para o arquivo
        
    Returns:
        bool: True se o arquivo for válido, False caso contrário
    """
    if not os.path.exists(file_path):
        print(f"Erro: O arquivo {file_path} não existe.")
        return False
        
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"Erro: O arquivo {file_path} está vazio (0 bytes).")
        return False
        
    print(f"Arquivo {file_path} existe e tem {file_size} bytes ({file_size/1024:.2f} KB)")
    
    # Tentar abrir o arquivo para verificar se está corrompido
    try:
        with open(file_path, 'rb') as f:
            header = f.read(10)  # Ler os primeiros bytes
            if len(header) < 10:
                print("Aviso: Arquivo muito pequeno, pode não conter áudio.")
            
            # Verificar se parece ser um arquivo MP3 (cabeçalho MP3 começa com ID3 ou 0xFF 0xFB)
            if file_path.lower().endswith('.mp3'):
                if not (header.startswith(b'ID3') or b'\xff\xfb' in header):
                    print("Aviso: Este não parece ser um arquivo MP3 válido.")
                    return False
    except Exception as e:
        print(f"Erro ao tentar ler o arquivo: {str(e)}")
        return False
        
    return True

def transcribe_with_google_speech(audio_path):
    """
    Alternativa para transcrição usando Google Speech Recognition via Speech Recognition library
    
    Args:
        audio_path (str): Caminho para o arquivo de áudio
        
    Returns:
        str: Texto transcrito ou mensagem de erro
    """
    try:
        import speech_recognition as sr
        print("Inicializando reconhecimento de fala...")
        
        r = sr.Recognizer()
        
        # Para arquivos MP3, precisamos convertê-los para WAV
        # já que speech_recognition suporta WAV diretamente
        print("Convertendo o áudio para formato compatível...")
        
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(audio_path)
            
            # Salvar temporariamente como WAV
            temp_wav = "temp_audio_file.wav"
            sound.export(temp_wav, format="wav")
            
            # Processar o arquivo WAV
            with sr.AudioFile(temp_wav) as source:
                print("Carregando o áudio...")
                audio_data = r.record(source)
                
                print("Realizando transcrição...")
                text = r.recognize_google(audio_data)
                
                # Limpar o arquivo temporário
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                    
                return text
                
        except ImportError:
            return "Para arquivos MP3, é necessário instalar a biblioteca pydub: pip install pydub"
        except Exception as e:
            return f"Erro na conversão/transcrição: {str(e)}"
            
    except ImportError:
        return "Biblioteca speech_recognition não encontrada. Instale com: pip install SpeechRecognition"

def transcribe_with_openai_whisper(audio_path):
    """
    Alternativa para transcrição usando Whisper da OpenAI via API
    
    Args:
        audio_path (str): Caminho para o arquivo de áudio
        
    Returns:
        str: Texto transcrito ou mensagem de erro
    """
    try:
        import openai
        
        print("Você tem uma chave API da OpenAI? (s/n)")
        response = input()
        
        if response.lower() == 's':
            api_key = input("Digite sua chave API da OpenAI: ")
            openai.api_key = api_key
            
            print("Enviando arquivo para a API Whisper da OpenAI...")
            
            try:
                with open(audio_path, 'rb') as audio_file:
                    result = openai.Audio.transcribe("whisper-1", audio_file)
                    return result["text"]
            except Exception as e:
                return f"Erro ao acessar a API da OpenAI: {str(e)}"
        else:
            return "Transcrição cancelada. É necessária uma chave da OpenAI."
    except ImportError:
        return "Biblioteca openai não encontrada. Instale com: pip install openai"

def main():
    if len(sys.argv) < 2:
        print("Uso: python transcrever_audio.py caminho/para/arquivo.mp3")
        return
    
    # Obter o caminho do arquivo de áudio do argumento da linha de comando
    audio_path = sys.argv[1]
    
    # Verificar o arquivo
    if not check_file(audio_path):
        print("\nO arquivo parece ter problemas. Deseja tentar alternativas? (s/n)")
        response = input()
        
        if response.lower() != 's':
            print("Operação cancelada.")
            return
            
        print("\nEscolha uma opção:")
        print("1. Tentar com Google Speech Recognition")
        print("2. Tentar com OpenAI Whisper (requer chave API)")
        print("3. Cancelar")
        
        choice = input("Digite o número da opção: ")
        
        if choice == "1":
            transcription = transcribe_with_google_speech(audio_path)
        elif choice == "2":
            transcription = transcribe_with_openai_whisper(audio_path)
        else:
            print("Operação cancelada.")
            return
    else:
        print("\nEscolha uma opção para transcrição:")
        print("1. AssemblyAI (requer upload)")
        print("2. Google Speech Recognition")
        print("3. OpenAI Whisper (requer chave API)")
        
        choice = input("Digite o número da opção: ")
        
        if choice == "1":
            print("Esta opção requer upload para a AssemblyAI. Deseja continuar? (s/n)")
            if input().lower() != 's':
                print("Operação cancelada.")
                return
                
            # Código de upload para AssemblyAI...
            print("Devido a problemas anteriores com o upload, esta opção está temporariamente indisponível.")
            return
        elif choice == "2":
            transcription = transcribe_with_google_speech(audio_path)
        elif choice == "3":
            transcription = transcribe_with_openai_whisper(audio_path)
        else:
            print("Opção inválida. Operação cancelada.")
            return
    
    # Criar o nome do arquivo de saída
    output_path = Path(audio_path).with_suffix('.txt')
    
    # Salvar a transcrição em um arquivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"\nTranscrição salva em: {output_path}")
    
    # Mostrar uma prévia da transcrição
    print("\nPrimeiras 150 caracteres da transcrição:")
    preview = transcription[:150] + "..." if len(transcription) > 150 else transcription
    print(preview)

if __name__ == "__main__":
    main()