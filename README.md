<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/profile-charts-dark.svg" />
  <img width="100%" alt="learning curve, language x domain, attention weights" src="assets/profile-charts-light.svg" />
</picture>

## 👋 Hi, I'm Miya

小型LLMを「自分のデータで、自分の用途に合わせて」育てることに取り組んでいる学生です。
モデルの中身を理解すること、そして作ったモデルを実際に動くアプリに組み込むことの両方を大事にしています。

## 🔭 What I'm doing now

### 🧠 Main: モデルの「中身」を理解する

**外部講義で、LLM・Physical AI・AIエンジニアリングを体系的に学んでいます**

| 分野 | 学んでいること |
| --- | --- |
| **LLM** | Transformer / Attentionの仕組み、事前学習とスケーリング則、SFT・LoRA・DPOによる追加学習、強化学習によるアラインメント、量子化、安全性、RAG・エージェント |
| **Physical AI** | 模倣学習と強化学習によるロボットの行動学習、VLA(画像と言語から行動を出す基盤モデル)、シミュレーション環境での実験 |
| **AI Engineering** | 学習したモデルの推論APIとしてのデプロイ、MLOps・CI/CDによる学習〜公開の自動化 |

- 最終課題では、QLoRAでSFTを行ったあとDPOで好みを学習させ、Base → SFT → DPO と段階ごとの性能差を自分で検証

**Transformerを0から自作する** — [Transformer0-1](https://github.com/miyamoto-gt/Transformer0-1)
- Attentionをnumpyで手書きし、PyTorchで小さなTransformerを組み立てる
- 一部を壊したり外したりして、挙動がどう変わるかを観察する
- 次は、自作Transformerに素数判定を学習させ、内部で何を学んでいるかを可視化する予定

### 🔧 Applied: 学んだことをアプリに組み込む
- 大きなモデルの出力を教師データにして、小型モデル(LFM2.5)をLoRAで学習させる — あいづち
- 3B規模のモデル単体の限界を測り、RAGで補う

### 🌱 Community
- LLM-jp に参加中

## 🎯 Goal

卒業までに、自分で学習したモデルを1つ公開し、他の人が使える状態にすること。

## 🚀 Projects

| Project | 何をしたか | 技術 |
| --- | --- | --- |
| [あいづち \| AIZUCHI](https://github.com/Hackit-2026/Team18) | 一人配信にAIが視聴者コメントを返すアプリ。基盤サイトを実装し、コメント生成用の小型モデルを自作データで学習中(hackit 2026 DMM賞) | Node.js / OpenAI API / LFM2.5 / Unsloth |
| [kz_2607](https://github.com/jphacks/kz_2607) | (ここに一言説明)。性格の軸ごとのLoRAによる生成と、自作Transformerによる判定モデルを担当(JPHACKS 2026) | PyTorch / LoRA |
| [Transformer0-1](https://github.com/miyamoto-gt/Transformer0-1) | AttentionのスクラッチからTransformerの自作まで | NumPy / PyTorch |
| [weather_model](https://github.com/miyamoto-gt/weather_model) + [weather_predict_site](https://github.com/miyamoto-gt/weather_predict_site) | 気象庁データで翌日の天気を予測するモデルを作り、API化してWebアプリに(個人開発) | pandas / statsmodels / FastAPI |
| [chatbot](https://github.com/miyamoto-gt/chatbot) | 機械学習の学習支援チャットボット。質問履歴から「学んだこと・復習すべきこと」を要約(個人開発) | Gemini API / Express |
**🤖 Models**
- 小型モデル:LFM2 / LFM2.5(Liquid AI)、Qwen3、sarashina2.2-3b
- 埋め込みモデル:Ruri(日本語の検索・RAG用)

**🔧 Training & Inference**
- 追加学習:LoRA / QLoRA、SFT、DPO
- ツール:Unsloth、Hugging Face、Ollama、mlx-lm、Google Colab
- 応用:RAG、教師モデルの出力を使った蒸留
## 🌱 Skills

<img alt="skills" src="https://skillicons.dev/icons?i=python,pytorch,fastapi,js,ts,nodejs,express,react,html,css,git,aws" />
