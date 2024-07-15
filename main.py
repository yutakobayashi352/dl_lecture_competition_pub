import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


def set_seed(seed):
    """
    このset_seed関数は、与えられたシード値を使用して、乱数ジェネレーターのシードを設定するためのものです。これにより、再現性のある結果を得ることができます。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    """
    このprocess_text関数は、与えられたテキストを処理して、より機械処理しやすい形式に変換するためのものです。まず、全ての文字を小文字に変換します。これにより、大文字と小文字の違いによるデータの不一致を防ぎます。
    次に、英語の数詞（'one', 'two', 'three'など）を対応する数字に変換します。これは、テキスト内の数値を一貫した形式に統一するために役立ちます。この変換は、num_word_to_digit辞書を使用して行われ、replaceメソッドを用いて各数詞をその数字に置き換えます。
    その後、正規表現を使用して、文中の孤立した小数点を削除し（数字の前後にないピリオド）、特定の冠詞（'a', 'an', 'the'）を削除します。これにより、テキストから不要な情報を取り除き、データの一貫性を高めます。
    短縮形の処理では、特定の短縮形を正しい形に修正します。例えば、"dont"を"don't"に変換します。これは、テキスト内の略語や短縮形を標準化するのに役立ちます。
    さらに、句読点をスペースに変換し、不要なスペースやカンマを整理します。これにより、テキストがより整った形で処理され、後続のテキスト解析や機械学習モデルへの入力として適切な状態になります。
    最後に、stripメソッドを使用して、テキストの先頭と末尾の余分なスペースを削除します。これにより、テキストの前後に不要な空白がなくなり、データのクリーンアップが完了します。
    この関数は、テキストデータの前処理において非常に有用であり、テキストベースのデータセットを解析や機械学習アルゴリズムに適した形に整形する際に役立ちます。
    """
    
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        """
        Pythonで書かれたクラスのコンストラクタ（__init__メソッド）です。このメソッドは、クラスのインスタンスが作成される際に自動的に呼び出され、インスタンス変数の初期化や必要なセットアップを行います。
        """
        
        # コンストラクタは、データフレームのパス(df_path)、画像ディレクトリ(image_dir)、画像の前処理を行うためのオプショナルな変換関数(transform)、そして回答を使用するかどうかを指定するブール値(answer)を引数として受け取ります。これらの引数は、インスタンス変数に保存され、クラスの他のメソッドからアクセスできるようになります。
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        # pandas.read_json関数を使用して、df_pathからJSON形式のデータを読み込み、DataFrameオブジェクトを作成します。このDataFrameは、画像ファイルのパス、質問(question)、回答(answers)を含んでいます。
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        # 質問文(question)と回答(answers)から単語を抽出し、それぞれを一意のインデックスにマッピングする辞書(question2idx、answer2idx)を作成します。これにより、テキストデータを数値データに変換し、機械学習モデルで扱いやすくします。また、逆変換用の辞書(idx2question、idx2answer)も作成し、インデックスから元の単語を取得できるようにします。
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．
        
        この__getitem__メソッドは、特定のデータセットクラスの一部として定義されており、指定されたインデックス(idx)に対応するデータ（画像、質問、回答）を取得するために使用されます。このメソッドは、PyTorchのDatasetクラスを継承したクラスでよく見られる実装パターンです。データセットから特定のアイテムを取得する際に、このメソッドが内部的に呼び出されます。
        
        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image) # self.transformを適用して前処理を行います。これにより、画像データがモデルの入力として適切な形式に変換されます。
        
        # 次に、質問文を処理します。質問文は文字列として格納されており、これをone-hot表現に変換することで、モデルが処理しやすい形式にします。one-hot表現では、質問文に含まれる各単語が、辞書(self.idx2question)に基づいてベクトルの特定の位置にマッピングされます。質問文に未知の単語が含まれる場合は、特別な"未知語"用の要素に1が設定されます。
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        question_words = self.df["question"][idx].split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換
            except KeyError:
                question[-1] = 1  # 未知語

        # さらに、回答データが存在する場合（self.answerがTrueの場合）、各回答者の回答を処理し、それらの回答IDのリストを作成します。そして、これらの回答から最頻値（最も一般的な回答）を計算し、そのIDを取得します。
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            # 最後に、処理された画像データ、質問データ、（存在する場合は）回答データと最頻値の回答IDを返します。これにより、モデルがこれらのデータを使用して学習や推論を行うことが可能になります。
            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    """
    このコードは、ビジュアル質問応答（VQA）タスクのための評価指標を実装しています。VQAタスクでは、画像に関する質問が与えられ、モデルはその質問に対する答えを生成する必要があります。この評価指標は、モデルが生成した予測と正解の一致度を測定するために使用されます。
    
    関数VQA_criterionは、予測された回答のバッチ（batch_pred）と、対応する正解のバッチ（batch_answers）を引数として受け取ります。各予測と正解のペアに対して、一致する回答の数をカウントし、その数に基づいて精度を計算します。
    
    内側のループ（for j in range(len(answers)):）では、各予測が正解のリスト内のどれだけの要素と一致するかをチェックします。ただし、自分自身との比較は除外されます（if i == j:）。一致する回答の数（num_match）に基づいて、その予測の精度を計算し（min(num_match / 3, 1)）、これを予測ごとの精度（acc）に加算します。ここで、min(num_match / 3, 1)は、一致する回答の数が3以上の場合でも精度を1として扱うためのものです。

    最終的に、全ての予測に対する平均精度（total_acc / len(batch_pred)）が計算され、返されます。この評価指標は、モデルがどれだけ正解に近い回答を生成できるかを示す一つの方法として利用できます。
    """
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        """
        この選択されたコードは、畳み込みニューラルネットワーク（CNN）の一部であるカスタム層の初期化関数を定義しています。この関数は、特にResNetのようなネットワークで使用される「ショートカット接続」または「スキップ接続」を含む層を構築するために設計されています。

        初期化関数は、入力チャネル数（in_channels）、出力チャネル数（out_channels）、およびストライド（stride）を引数として受け取ります。strideはデフォルトで1に設定されています。この関数は、まずsuper().__init__()を呼び出して、基底クラスのコンストラクタを実行します。

        次に、2つの畳み込み層（conv1とconv2）と、それぞれに対応するバッチ正規化層（bn1とbn2）を定義します。これらの層は、入力データに対して畳み込み演算を行い、モデルの学習能力を高めるために使用されます。conv1層は、指定されたstrideを使用していますが、conv2層は常にstride=1を使用します。これは、conv1でサイズを変更する可能性があるため、conv2でサイズを保持するためです。

        ReLU活性化関数も定義されており、inplace=Trueパラメータにより、計算効率を向上させるために入力を直接変更します。

        最後に、shortcutという名前のシーケンシャルコンテナが定義されています。これは、入力と出力のチャネル数が異なる場合、またはストライドが1ではない場合に、入力を変換して出力とサイズを合わせるために使用されます。これにより、入力と出力を直接加算することが可能になり、ショートカット接続が実現されます。shortcutは、条件に応じて1x1の畳み込み層とバッチ正規化層を含むシーケンシャルコンテナに設定されます。これにより、入力のチャネル数やサイズを、出力層のそれと一致させることができます。

        このコードは、深いネットワークでの勾配消失問題を軽減し、より深いネットワークの訓練を可能にするResNetの重要な特徴であるショートカット接続の実装を示しています。
        """
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        このコードは、ニューラルネットワークの一部であるResidual Blockの実装を示しています。Residual Blocksは、深いニューラルネットワークを訓練する際に生じる勾配消失問題を軽減するために設計されました。このコードは、PyTorchのようなディープラーニングフレームワークで使用されることが想定されています。

        最初に、residual変数に入力xを保存します。これは後で、出力に入力を直接加算するために使用されます。これが「ショートカット接続」または「スキップ接続」と呼ばれるものです。

        次に、xは最初の畳み込み層self.conv1を通過し、その後バッチ正規化層self.bn1を通過します。バッチ正規化は、ネットワークの訓練を安定させ、加速させるために使用されます。その後、ReLU活性化関数self.reluを適用します。ReLUは、非線形性を導入し、モデルがより複雑な関数を学習できるようにします。

        その結果得られたoutは、2番目の畳み込み層self.conv2とバッチ正規化層self.bn2を順に通過します。

        次に、self.shortcut(residual)を使用して、元の入力x（ここではresidualとして保存されている）を現在の出力outに加算します。これにより、ネットワークが学習する際に、入力信号がネットワークの深い層を通過するときにその強度を保持できるようになります。

        最後に、ReLU活性化関数をもう一度適用して、最終的な出力を得ます。そして、この出力を関数の呼び出し元に返します。

        このコードスニペットは、深いニューラルネットワークの中で情報がどのように処理され、変換されるかを示す良い例です。ショートカット接続の使用は、深いネットワークの訓練を改善する上で重要な役割を果たします。
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        """
        この選択されたコードは、深層学習で広く使用されているResNet（Residual Network）アーキテクチャの実装の一部です。ResNetクラスは、PyTorchのnn.Moduleクラスを継承しており、これによりカスタムニューラルネットワークモデルを定義できます。このクラスのコンストラクタ__init__では、モデルの初期化が行われ、畳み込み層、バッチ正規化層、ReLU活性化関数、プーリング層、そして複数のレイヤーを生成するための_make_layerメソッドの呼び出しが含まれています。
        """
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        """
        _make_layerメソッドは、指定されたブロックタイプとブロック数を使用して、ネットワークの中間層を動的に生成します。このメソッドは、畳み込みブロックのシーケンスを作成し、それらをnn.Sequentialコンテナにラップして、順序付けられたモジュールのシーケンスとして返します。各ブロックは、入力
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    # このコードは、視覚質問応答(VQA)モデルの定義を示しています。VQAは、画像とそれに関する質問が与えられたときに、適切な回答を生成するタスクです。このモデルは、PyTorchのnn.Moduleクラスを継承しており、ニューラルネットワークモデルを構築するための基本クラスです。
    def __init__(self, vocab_size: int, n_answer: int):
        # __init__メソッドでは、モデルの構造を定義しています。まず、super().__init__()を呼び出して、基底クラスのコンストラクタを初期化します。次に、ResNet18関数を使用して、画像特徴量を抽出するためのResNet18モデルをself.resnetに割り当てます。nn.Linear(vocab_size, 512)を使用して、質問テキストの特徴量をエンコードするための線形変換をself.text_encoderに設定します。最後に、self.fcには、画像特徴量とテキスト特徴量を結合した後に、最終的な回答を生成するための全結合層のシーケンスが設定されています。
        super().__init__()
        self.resnet = ResNet18() ### ここでResNetを利用する
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        # forwardメソッドでは、モデルの順伝播を定義しています。まず、self.resnet(image)を使用して画像から特徴量を抽出し、self.text_encoder(question)を使用して質問テキストから特徴量を抽出します。次に、torch.cat関数を使用して、これらの特徴量を結合します。この結合された特徴量は、self.fcを通過して最終的な回答の予測に使用されます。forwardメソッドの最後に、この予測結果を返します。
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    """
    このコードは、ビジュアル質問応答（VQA）モデルの学習プロセスを実装しています。学習プロセスでは、モデルが画像と質問を入力として受け取り、それに対する答えを予測する能力を向上させることが目的です。この関数は、モデル、データローダー、オプティマイザー、損失関数、およびデバイス（CPUまたはGPU）を引数として受け取ります。

    学習プロセスは、まずモデルを訓練モードに設定し、損失と精度を追跡するための変数を初期化します。次に、データローダーからバッチごとに画像、質問、答え、および最頻値の答え（mode_answer）を取得し、それらを指定されたデバイスに移動します。モデルは画像と質問を入力として受け取り、答えの予測（pred）を生成します。この予測と最頻値の答えを用いて損失が計算され、バックプロパゲーションを通じてモデルの重みが更新されます。

    さらに、この関数は2種類の精度を計算します。1つ目は、VQA_criterion関数を使用して計算されるVQA精度で、予測された答えと実際の答えの一致度を測定します。2つ目は、単純精度（simple accuracy）で、予測された最も可能性の高い答えが最頻値の答えと一致するかどうかを測定します。これらの精度は、学習プロセスの進行状況を評価するために使用されます。

    最後に、この関数は、平均損失、VQA精度、単純精度、および学習にかかった時間を返します。これにより、学習プロセスの効率と効果を評価することができます。
    """
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    """
    このコードは、ビジュアル質問応答（VQA）モデルの評価プロセスを実装しています。評価プロセスでは、学習済みのモデルが新しいデータに対してどの程度うまく機能するかを測定します。この関数は、モデル、データローダー、オプティマイザー、損失関数、およびデバイス（CPUまたはGPU）を引数として受け取ります。

    評価を開始する前に、model.eval()を呼び出してモデルを評価モードに設定します。これにより、ドロップアウトやバッチ正規化などの学習時の挙動を変更するレイヤーが評価時の挙動に切り替わります。

    評価プロセスでは、データローダーからバッチごとに画像、質問、答え、および最頻値の答え（mode_answer）を取得し、それらを指定されたデバイスに移動します。モデルは画像と質問を入力として受け取り、答えの予測（pred）を生成します。この予測と最頻値の答えを用いて損失が計算されます。

    さらに、2種類の精度が計算されます。1つ目は、VQA_criterion関数を使用して計算されるVQA精度で、予測された答えと実際の答えの一致度を測定します。2つ目は、単純精度（simple accuracy）で、予測された最も可能性の高い答えが最頻値の答えと一致するかどうかを測定します。

    最後に、この関数は、平均損失、VQA精度、単純精度、および評価にかかった時間を返します。これにより、モデルの性能を定量的に評価することができます。評価プロセスは、モデルの改善点を特定し、さらなるチューニングを行うための重要なステップです。
    """
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def main():
    """
    このコードは、PyTorchを使用して視覚質問応答(VQA)モデルを訓練し、評価するプロセスを実装しています。VQAタスクは、画像とそれに関連する質問が与えられたときに、適切な回答を生成することを目指します。

    このコードは、データの前処理からモデルの訓練、評価、そして結果の保存まで、VQAタスクの一般的なワークフローをカバーしています。
    """
    # deviceの設定
    # まず、乱数のシードを設定して再現性を確保し、使用するデバイスをCUDA対応GPUが利用可能かどうかに基づいて選択します。これにより、計算が高速化されます。

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # dataloader / model
    # 次に、訓練用とテスト用のデータセットを準備します。これには、画像のリサイズとテンソルへの変換を含む前処理が含まれます。VQADatasetクラスは、これらのデータセットを管理し、DataLoaderを使用してバッチ処理とデータのシャッフルを行います。
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルはVQAModelクラスのインスタンスで、質問の語彙サイズと回答の数に基づいて初期化されます。このモデルは選択されたデバイスに移動されます。
    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    # 訓練プロセスでは、クロスエントロピー損失とAdamオプティマイザーが使用されます。訓練は1エポックだけ行われ、各エポックで訓練データを使用してモデルを更新し、訓練の損失と精度を計算します。
    num_epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    # 最後に、モデルを評価モードに切り替え、テストデータセット上でモデルを評価します。各画像と質問のペアに対して、モデルは最も可能性の高い回答を予測し、これらの予測を使用して提出用ファイルを作成します。モデルの状態と予測結果はそれぞれmodel.pthとsubmission.npyに保存されます。    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()



# VQAの精度を向上させるために、以下の改善を行います：
# データ拡張：学習データのバリエーションを増やすため、画像のデータ拡張を追加します。
#ハイパーパラメータのチューニング：学習率やエポック数などのハイパーパラメータを最適化します。
#モデルアーキテクチャの改善：ResNet50など、より深いモデルを使用します。
#適切な損失関数の選択：Focal Lossなど、バランスの取れた損失関数を使用します。
#アンサンブル学習：複数のモデルを学習し、それらの予測を平均化することで精度を向上させます。