# 💬 Fine-Tuning a Large Language Model on the Banking77 Dataset
**👨‍💻 Author:** Sinon Lobo  
**📘 Course:** Advanced NLP / LLM Fine-Tuning  
**📅 Date:** October 2025  

---

## 🧠 Project Overview
This project explores **parameter-efficient fine-tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** on the **DistilBERT** model to perform **intent classification** on the **Banking77 dataset**.  

The primary goal was to adapt a pre-trained transformer to accurately understand **customer banking queries** such as *“I lost my card”* or *“How can I reset my PIN?”* — while significantly reducing computational cost.

🚀 Using LoRA, we updated **less than 1%** of DistilBERT’s parameters and achieved a remarkable **94% accuracy**, demonstrating that lightweight fine-tuning methods can rival traditional full-model training in performance while being much more efficient.

---

## 🎯 Project Objectives
✅ Fine-tune a pre-trained **DistilBERT** model using **LoRA**  
✅ Classify **77 distinct banking-related intents** from short text queries  
✅ Leverage **PEFT** for efficient, low-resource fine-tuning  
✅ Achieve high accuracy and F1-score with minimal parameter updates  
✅ Build an interactive **Gradio-based web interface** for real-time inference  

---

## 📊 Dataset Overview
**Dataset:** [Banking77 (Hugging Face)](https://huggingface.co/datasets/banking77)  
**Task Type:** Intent Classification  

The **Banking77 dataset** is a benchmark dataset in conversational AI, designed for **fine-grained intent detection** in customer support scenarios.  

### 📁 Dataset Details
- **Classes:** 77 user intent categories  
- **Training Samples:** ~10,000  
- **Testing Samples:** ~3,000  
- **Input:** User query text  
- **Output:** Intent label (category)  

### 🧩 Example Entries
| User Query | Intent Label |
|-------------|--------------|
| “How do I activate my card?” | `activate_my_card` |
| “I forgot my PIN number.” | `forgot_pin` |
| “When will my card arrive?” | `card_arrival` |

Preprocessing included **tokenization**, **padding**, **truncation**, and **label encoding** — ensuring compatibility with the Hugging Face Transformers pipeline.

---

## 🧩 Model Architecture
### 🧱 Base Model: `distilbert-base-uncased`
DistilBERT is a **lightweight, faster version of BERT** that retains about **97%** of its performance while being:
- ⚡ 60% faster in inference
- 💾 40% smaller in size
- 🧠 Ideal for limited compute environments  

It uses **6 transformer encoder layers** (compared to 12 in BERT) and was trained using **knowledge distillation** to compress language understanding into a smaller model.

---

### ⚙️ Why DistilBERT + LoRA?
| Feature | Benefit |
|----------|----------|
| Lightweight Transformer | Perfect for small-scale fine-tuning |
| LoRA Adaptation | Updates <1% of weights efficiently |
| PEFT Library | Simplifies LoRA integration and management |
| Faster Convergence | Less training time with similar accuracy |

---

### 🧮 LoRA Configuration
LoRA introduces trainable low-rank matrices (A, B) into attention layers, replacing full weight updates with efficient transformations.

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                 # Rank of low-rank matrices
    lora_alpha=16,       # Scaling factor
    lora_dropout=0.1,    # Dropout rate
)
