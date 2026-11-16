# IA-SugestaoPratos: Extração de Intenção de Busca em Delivery

Este projeto mostra como utilizar small language models (SLMs) e fine-tuning para resolver um problema no setor de food delivery: transformar buscas informais e cheias de gírias em uma estrutura JSON padronizada

O modelo recebe a linguagem natural de um usuário (com erros de digitação, regionalismos e restrições alimentares) e devolve as chaves para um sistema de busca tradicional processar

**Modelo Publicado:** https://huggingface.co/Pukanski/sugerePrato-intent-qwen-1.5b-lora

---

## Arquitetura e Stack
- **Modelo Base:** `Qwen2.5-1.5B-Instruct`
- **Técnica de Treinamento:** **QLoRA** (Quantized Low-Rank Adaptation) em 4-bits
- **Bibliotecas:** `Unsloth`, `Hugging Face Transformers`, `TRL` e `PEFT`.

---

## Como Usar

Para testar diretamente no python utilizando a biblioteca `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

modelo_id = "Pukanski/sugerePrato-intent-qwen-1.5b-lora"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
model = AutoModelForCausalLM.from_pretrained(modelo_id)

prompt = "mano manda um podrão caprichado sem milho"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

print(tokenizer.decode(outputs[0]))
```
---

## Estratégia de Dados: Geração Sintética
Como datasets públicos de logs reais de delivery em português não estão disponíveis, a solução foi a geração de dados sintéticos estruturados

Utilizando LLMs (Claude/Gemini), criei um dataset de ~160 exemplos dividido em categorias do mundo real:
1. **Gírias e Regionalismos:** "podrão", "xis", "larica".
2. **Restrições Dietéticas:** "sem lactose", "vegano", "zero açúcar".
3. **Filtros e Ordenação:** "menos de 30 conto", "que chegue rápido".
4. **Erros de Digitação e Abreviações:** "pitza s/ cebola", "lanxe".

Os dados foram processados com Python/Pandas e divididos via `train_test_split` (80% treino / 20% teste) no formato `.jsonl`.

---

## Desafios Encontrados e Soluções de Engenharia

Durante o desenvolvimento, identifiquei comportamentos indesejados comuns em LLMs que precisaram ser corrigidos na arquitetura e no código:

1. **Alucinação Semântica com Gírias**
   - *Problema:* Durante as primeiras épocas, o modelo não compreendia gírias locais. Ao receber a palavra "podrão", o modelo tentava "corrigir" o texto e alucinava extraindo algo como *"mão de manjericão"* no JSON gerado.
   - *Solução:* Aumentei o número de épocas de treinamento para forçar o ajuste dos pesos as nuances do novo vocabulário. Adicionei também o token de fim de sequência no dataset para que o modelo aprendesse onde parar a geração

2. **Sensibilidade Espacial no Tokenizador**
   - *Problema:* A formatação do prompt de inferência possuía espaços em branco acidentais no início de algumas linhas. LLMs pequenos são extremamente sensíveis a espaçamentos, o que alterava completamente a representação dos *tokens* e quebrava o raciocínio.
   - *Solução:* Refatoração da string do prompt, garantindo alinhamento total com a formatação exata que o modelo viu durante a fase de treino

3. **Falta de Determinismo na Geração do JSON**
   - *Problema:* Por padrão, o modelo utilizava sampling probabilístico, tornando-se "criativo" demais na hora de preencher chaves nulas ou Booleanas.
   - *Solução:* Na etapa de inferência, desabilitei a temperatura e o sampling (`do_sample=False`), forçando uma geração determinística focada na extração das entidades.

---

## Exemplo de Inferência (Antes vs Depois)

**Input do Usuário:**  
*"to varado de fome manda um podrão caprichado sem milho e sem ervilha que passe vr"*

**Antes do Fine-Tuning:**
```json
{
  "prato_ou_culinaria": "mão de manjericão", 
  "restricoes_exclusao": ["milho", "erva-doce"], 
  "restricoes_dieta": [], 
  "preco_maximo": null, 
  "ordenar_por": "tempo_de_entrega"
}
```

**Depois do Fine-Tuning:**
```json
{
  "prato_ou_culinaria": "podrão", 
  "restricoes_exclusao": ["milho", "ervilha"], 
  "restricoes_dieta": [], 
  "preco_maximo": null, 
  "ordenar_por": "relevancia"
}
```

---

## Próximos Passos

Este projeto é uma prova de conceito. Para escalar este serviço em um ambiente de produção real lidando com milhões de requisições, os próximos passos seriam:

1. **Aumento Massivo de Dados:**
   - Escalar o dataset, focando especialmente no longo padrão de intenções vagas e erros gramaticais pesados, aumentando a resiliência do modelo
2. **Pipeline de Avaliação Automatizada:**
   - Implementar scripts utilizando frameworks como `Ragas` ou `TruLens` para calcular o *Exact Match Score* do JSON e avaliar o grau de alucinação de forma automatizada a cada novo checkpoint de treino.
3. **Deploy Otimizado:**
   - Servir o modelo ajustado utilizando motores de inferência de alta performance para reduzir a latência de extração
