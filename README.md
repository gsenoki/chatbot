# Chatbot RAG

Este repositório contém a implementação de um chatbot baseado em **Retrieval-Augmented Generation (RAG)**, projetado para responder perguntas sobre o **Vestibular Unicamp 2025**.

## Instalação

### 1. Clone o repositório
Clone este repositório para sua máquina local usando o comando abaixo:
```bash
git clone https://github.com/gsenoki/chatbot.git
cd SEU_REPOSITORIO
```
### 2. Instale as dependencias
Certifique-se de que você possui Python 3.7 ou superior instalado. Em seguida, instale as dependências necessárias:
```bash
pip install -r requirements.txt
```
### 3. Configure a chave API de ChatGroq
Este projeto utiliza a API ChatGroq para acessar o modelo de linguagem. Você precisa de uma chave de API para utilizar a funcionalidade do chatbot.
1. Obtenha sua chave no site oficial da [Groq](https://groq.com/).
2. Cole a chave no terminal quando aparecer "Password"

## Codigos
Nesse repositorio temos 3 arquivos python
1. chatbot_streamlit.py que é usado pelo streamlit cloud, [link do app](https://chatbot-amgjmmns84fufvxerh4pkt.streamlit.app/)
2. chatbot.py o chatbot de terminal, roda indefinitivamente respondendo perguntas
3. chatbot_test.py que roda os test dataset

### 1. Execute o chatbot
Inicie o chatbot com o seguinte comando:
```bash
python chatbot.py
python chatbot_test.py
```
### 2. Interaja com o chatbot
O chatbot estará disponível para responder perguntas. Basta digitar uma pergunta relacionada ao Vestibular Unicamp 2025.
