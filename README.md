# 🍊 Detecção de Laranjas em Laranjeiras: SSD MobileNet V2 (Trabalho de Mestrado)

Este repositório documenta o trabalho de mestrado sobre a **Detecção de Laranjas usando Single-Shot Multibox Detector com Arquitetura MobileNet V2 (SSDLite)**. [cite_start]O projeto tem como objetivo substituir os métodos manuais de contagem de frutos (como a derriça) por uma ferramenta mais **rápida, barata e eficaz** para a estimativa da safra anual[cite: 34].

[cite_start]O trabalho foi desenvolvido na **Universidade Estadual de Campinas (UNICAMP)** [cite: 3] [cite_start]no Instituto de Matemática, Estatística e Computação Científica (IMECC)[cite: 7].

## ⚠️ Aviso Importante sobre os Dados

[cite_start]O banco de dados de **3028 imagens** de dimensão $(416 \times 416)$ [cite: 91] [cite_start]foi fornecido pelo **Fundecitrus** (Fundo de Defesa da Citricultura) [cite: 90] e não pode ser distribuído.

**Você está livre para usar toda a metodologia e o código** aqui presente para aplicar a detecção de objetos em seus próprios conjuntos de dados (outras frutas, objetos, etc.). [cite_start]As configurações de treinamento foram otimizadas e podem ser um excelente ponto de partida para outros projetos[cite: 468].

* [cite_start]**Contato:** m.chiqueto@usp.br (E-mail USP) / m264864@dac.unicamp.br [cite: 6]

---

## 🚀 Resultados e Desempenho

[cite_start]O modelo **SSDLite (MobileNetV2)** demonstrou um ótimo desempenho, superando o modelo SSD tradicional (com VGG) e apresentando um custo computacional muito baixo[cite: 467].

| Modelo | AP@0.5 | Parâmetros | Tamanho do Modelo |
| :--- | :--- | :--- | :--- |
| **SSDLite (MobileNetV2)** | **0.88** | 4 M | **4,8 MB** |
| SSD (VGG) | 0.814 | 27 M | 182 MB |

[cite_start]*Tabela: Comparação entre SSDLite e SSD VGG (adaptada da Apresentação)[cite: 444].*

**Destaques da Performance:**

* [cite_start]**Precisão:** O valor final de $AP@0.5 = 0.88$ indica um ótimo desempenho de detecção[cite: 418].
* [cite_start]**Eficiência:** O modelo final exige apenas **4,8 MB** de memória, o que viabiliza seu uso em dispositivos móveis e permite treinamento com outros bancos de dados sem grandes requisitos de processamento[cite: 468].
* [cite_start]**Desafio:** O desempenho em **objetos pequenos** foi relativamente baixo em comparação com objetos médios e grandes, com $AP@Small$ com valor máximo em torno de $0.4$[cite: 419, 420].

---

## 💻 Metodologia de Detecção

### 1. Arquitetura da Rede (MobileNet V2)

[cite_start]O trabalho utilizou o detector **SSD** com o extrator de características **MobileNet V2**[cite: 55].

* [cite_start]**Convoluções Separáveis Profundas:** A característica mais distintiva da MobileNetV2 é o uso de blocos residuais invertidos com gargalo linear[cite: 227].
* [cite_start]**Desenvolvimento:** A MobileNet V2 foi desenvolvida especificamente para dispositivos móveis[cite: 229].

### 2. Configuração do Treinamento

* [cite_start]**Transferência de Aprendizado:** Os pesos da rede foram inicializados com aqueles da MobileNet V2 treinada com o banco de dados Microsoft COCO[cite: 402].
* [cite_start]**Regularização:** Foram usadas técnicas de regularização, incluindo `dropout` e Regularização de Tikhonov[cite: 403].
* [cite_start]**Aumento de Dados:** Aplicação de **"Corte Aleatório"** e **"Espelhamento Horizontal"** para gerar imagens de treinamento[cite: 404, 405].
* [cite_start]**Função Custo:** A perda total é dada pela soma ponderada da perda de classificação ($L_{conf}$) e a perda de localização ($L_{loc}$)[cite: 341].
    $$L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc}(x,l,g))$$
    * [cite_start]$L_{loc}$ é baseada na função Smooth-$L_{1}$[cite: 350].
    * [cite_start]$L_{conf}$ é a Softmax sobre as $c$ classes[cite: 360].

---

### Passos Chave (Resumo do Código)

1.  **Preparação:** Instalar as dependências, clonar o repositório `tensorflow/models` e montar o Drive.
2.  **Dados:** Adicione **suas próprias imagens** e anotações na estrutura `data/`. Execute a conversão `xml_to_csv` e a criação dos `.record` e `label_map.pbtxt`.
3.  **Configuração:** Baixar o modelo pré-treinado (e.g., `ssd_mobilenet_v2_320x320_coco17_tpu-8`) e editar o arquivo `.config` para apontar os caminhos corretos para os seus arquivos.
4.  **Treinamento:** Execute o `model_main_tf2.py`:
    ```bash
    !python /mydrive/Object_Detection/ssd/models/research/object_detection/model_main_tf2.py \
        --pipeline_config_path={pipeline_config} \
        --model_dir={model_dir_config} \
        --alsologtostderr \
        --eval_on_train_data=True
    ```
5.  **Exportação:** Exporte o `saved_model` final, utilizando o último *checkpoint* treinado, para obter o modelo final de inferência.

---

Se tiver dúvidas sobre a implementação, sinta-se à vontade para entrar em contato através do e-mail m.chiqueto@usp.br.
