# exampleについて

提出する評価用データの例，および，関連ツールを提供します．

## 構成

- create_stp_test_data.py
- evaluation.py
- ground-truth/
- submission/
- test-data/

### create_stp_test_data.py

テストデータを作成する．tracking.csv, event.csv, rcg.gz, rcl.gz のファイル群が置かれているディレクトリからタスクの条件に一致するゴールシーンを抽出し，ゴールシーンから30フレーム前までのデータを別ディレクトリへ出力します．
提供しているデータセットからテスト用データを作成することができます．
```
python create_stp_test_data <INPUT_DIR> <OUTPUT_DIR>
```

### evaluation.py

モデルが出力した予測データを正解データと比較し，ゴールシーンとなる最終フレームでの距離誤差の平均値を求めます．距離を測る対象となるのは，左チームの選手11体とボールです．右チームの選手は対象としません．
```
python evaluation.py --gt <GROUND_TRUTH_DIR> --input <INPUT_DATA_DIR> --submit <SUBMIT_DATA_DIR>
```
- GROUND_TRUTH_DIR: 正解データの.tracking.csvファイルが収められているディレクトリパス
- INPUT_DATA_DIR: テストデータとなる.tracking.csvが収められているディレクトリパス
- SUBMIT_DATA_DIR: モデルが出力した予測結果の.tracking.csvファイルが収められているディレクトリパス

実行に成功すると，```Endpoint Error: 数値``` という行が出力されます．この数値は，各試合での距離誤差平均値を全試合分合わせた平均値です．この数値を0に近づけることがタスクの目標です．

注意事項
- 各ディレクトリ以下の ".tracking.csv" ファイルが読み込まれます．対応するファイルの名前が一致していなければなりません．
- 予測結果の.tracking.csvには，フレームのインデックス，左チームの選手の位置座標，ボールの位置座標 が含まれていなければなりません．
- 予測結果の.tracking.csvのフレームインデックスは，対応する入力データの最終フレーム+1から開始しなければなりません．
- 予測結果の.tracking.csvに含まれるレコード数は30でなければなりません．ただし，評価に用いられるのは30番目のレコード（実際の試合ではゴールシーンのフレーム）のみです．

###  ground-truth/
1試合分の完全なログファイルです．

### test-data/
条件を満たすゴールシーンから30フレーム前までのデータのみを残したログファイルです．

### submission/
提出用データの例です．test-dataの最終フレームから30フレーム先までの インデックス，全選手の位置座標，ボールの位置座標 を.tracking.csvと同様のカラム名で書き出しています．提出データには，インデックス，左チームの選手の位置座標，ボールの位置座標 が必ず含まれていなければなりません．

## 評価の実行例
example以下のファイルを使って評価値を出力する例:
```
python evaluation.py --gt ./example/ground-truth --input ./example/test-data --submit ./example/submission
```