# Rank and Reconstitute

We use CoBERT from the work [Towards Persona-Based Empathetic Conversational Models](https://arxiv.org/abs/2004.12316)" (EMNLP 2020). The code depends on PyTorch (>=v1.0) and [transformers]((https://github.com/huggingface/transformers)) (>=v2.3).

makeRankingdata.py --> We transform elements of Optimized Persona-Chat for CoBERT's input structure.<br>
CoBERT_rank_out.py<br>
--> (1) Finetune CoBERT on Persona-Chat<br>
&emsp;(2) Rank candidates generated from the Control part by using CoBERT<br>
&emsp;(3) Reconstitute the top 3 responses into the Persona-Chat with COMET expansion<br>

CoBERT_config.json --> Set parameters (e,g. k is the number of selected syntehtic gold labels)
