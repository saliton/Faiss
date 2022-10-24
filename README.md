[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soliton-Analytics-Team/Faiss/blob/main/Faiss.ipynb)

# Faiss-gpuはどれぐらい速いのかcolabで試してみた

Googleから[ScaNN](https://github.com/google-research/google-research/tree/master/scann) (Scalable Nearest Neighbors)というベクトル近似近傍検索が出て、速さを売りにしています。確かに[ベンチマーク](http://ann-benchmarks.com)でも結果がでています。ただ、このベンチマーク、CPUオンリーで、GPUを使う近傍検索との比較がありません。GPUが使えるといえば、[Faiss](https://github.com/facebookresearch/faiss)ですね。というわけで、早速、GPUが使えるcolabで測定してみましょう。
結論を先に言うと、GPUすごく速いです。

と、その前に、ランタイムはまだGPUにしないでください。途中で切り替えないとうまく実行できません。最初はランタイムNoneで進めてください。

## データの取得

評価の対象とするデータは[ann-benchmarks](http://ann-benchmarks.com/)のglove-100-angularを使います。これ、Faissの中の人によると[ScaNNに有利なデータ分布](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors#4-bit-pq-comparison-with-scann)だそうなのですが、まあ、いいでしょう。

```shell
!wget http://ann-benchmarks.com/glove-100-angular.hdf5
```

    --2022-10-20 10:33:33--  http://ann-benchmarks.com/glove-100-angular.hdf5
    Resolving ann-benchmarks.com (ann-benchmarks.com)... 54.231.201.245
    Connecting to ann-benchmarks.com (ann-benchmarks.com)|54.231.201.245|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 485413888 (463M) [binary/octet-stream]
    Saving to: ‘glove-100-angular.hdf5’

    glove-100-angular.h 100%[===================>] 462.93M  12.4MB/s    in 38s

    2022-10-20 10:34:12 (12.2 MB/s) - ‘glove-100-angular.hdf5’ saved [485413888/485413888]

検索対象のデータは100次元で約100万件、クエリーデータは1万件です。neighborsに正解が入ります。

```python
import numpy as np
import time
import h5py

glove_h5py = h5py.File('glove-100-angular.hdf5')
dataset = glove_h5py['train']
queries = glove_h5py['test']
neighbors = glove_h5py['neighbors']
print("dataset", dataset.shape)
print("queries", queries.shape)
print("true_neighbors", neighbors.shape)
```

    dataset (1183514, 100)
    queries (10000, 100)
    true_neighbors (10000, 100)

なぜかデータを正規化しておかないとScaNNで上手く動きません。この辺りはよく分かっていませんが、ここでは追求せず先に進みます。

```python
normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
```

再現率を計算する関数を作っておきましょう。

```python
def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size
```

## ScaNN (CPU)

Faissと対比するため、先にScaNNで測定します。まずはインストール

```python
!pip install scann --quiet
import scann
```

    |████████████████████████████████| 10.4 MB 5.3 MB/s
    |████████████████████████████████| 578.0 MB 14 kB/s
    |████████████████████████████████| 438 kB 69.1 MB/s
    |████████████████████████████████| 1.7 MB 45.3 MB/s
    |████████████████████████████████| 5.9 MB 35.6 MB/s

最初に総当たりのモデルを作成します。これは一瞬ですね。

```shell
%%time
scann_brute = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").score_brute_force().build()
```

    CPU times: user 253 ms, sys: 495 ms, total: 748 ms
    Wall time: 739 ms

処理時間37秒。あたりまえですが、再現率100%。上手く処理できていることが確認できました。

```python
start = time.time()
scann_brute_ans = scann_brute.search_batched(queries, 10)
print(time.time() - start, "sec")
compute_recall(scann_brute_ans[0], neighbors[:, :10])
```

    37.05260634422302 sec
    1.0

次に、近似で検索速度重視のモデルを作ります。これは1分半ほど時間がかかります。

```shell
%%time
scann_searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()
```

    CPU times: user 2min 34s, sys: 1.43 s, total: 2min 36s
    Wall time: 1min 26s

処理時間は3.2秒、再現率は90%でした。

```python
start = time.time()
I, D = scann_searcher.search_batched(queries)
print(time.time() - start, "sec")
compute_recall(I, neighbors[:, :10])
```

    3.2482008934020996 sec
    0.90015

leaves_to_searchを指定すると、検索打ち切りまでの範囲を広げられるようです。これで処理時間が3.4秒。再現率が92%です。

```python
start = time.time()
scann_searcher_ans = scann_searcher.search_batched(queries, leaves_to_search=150)
print(time.time() - start, "sec")
compute_recall(scann_searcher_ans[0], neighbors[:, :10])
```

    3.4121041297912598 sec
    0.92392

pre_reorder_num_neighborsを設定しても、同様な効果があるそうです。処理時間が4.2秒。再現率が93%です。

```python
start = time.time()
scann_searcher_ans = scann_searcher.search_batched(queries, leaves_to_search=150, pre_reorder_num_neighbors=250)
print(time.time() - start, "sec")
compute_recall(scann_searcher_ans[0], neighbors[:, :10])
```

    4.178981065750122 sec
    0.93145

## Faiss (CPU)

CPU版のFaissをインストールします。ランタイムのタイプをGPUにしていると、このインストールが失敗します。

```shell
!apt install libomp-dev
!pip install faiss -U
import faiss
```

    Reading package lists... Done
    Building dependency tree
    ・
    ・
    ・
    Installing collected packages: faiss
    Successfully installed faiss-1.5.3

総当たり用のモデルを作成します。

```python
%%time
index = faiss.IndexFlatIP(100)
index.add(normalized_dataset)
```

    CPU times: user 342 ms, sys: 121 ms, total: 463 ms
    Wall time: 466 ms

総当たりで検索します。再現率100%で処理時間は87秒です。処理自体は上手く行っています。ただ処理時間はScaNNの2倍です。ScaNNはCPUに特化しているためか、実装に注力しているようです。

```python
start = time.time()
faiss_brute_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_brute_ans[1], neighbors[:, :10])
```

    86.79835033416748 sec
    1.0

チュートリアルに記載されていたIndexIVFFlatというモデルを作ります。

```python
%%time
nlist = 100
quantizer = faiss.IndexFlatIP(100)
index = faiss.IndexIVFFlat(quantizer, 100, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(normalized_dataset)
index.add(normalized_dataset)
```

    CPU times: user 3.9 s, sys: 1.04 s, total: 4.94 s
    Wall time: 3.04 s

デフォルトの状態での検索は、5.3秒で再現率は53%です。

```python
start = time.time()
faiss_searcher_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    5.289741277694702 sec
    0.53363

検索範囲を広げると再現率は83%になりましたが、処理時間が27.6秒です。

```python
index.nprobe = 5
start = time.time()
faiss_searcher_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    27.606809616088867 sec
    0.82666

```python
index.nprobe = 10
start = time.time()
faiss_searcher_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    48.293612003326416 sec
    0.90237

```python
index.nprobe = 15
start = time.time()
faiss_searcher_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    78.04950618743896 sec
    0.93486

```python
index.nprobe = 20
start = time.time()
faiss_searcher_ans = index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    96.26129055023193 sec
    0.95453

検索対象の向き不向きがあるとはいえ、CPUではScaNNのほうがFaissより高速なようです。同程度の再現率で10~20倍もの処理時間の差がある感じです。

## Faiss (GPU)

ここで、ランタイムのタイプをGPUに切り替えてください。切り替えるとランタイムが再起動してしまいます。再起動したら記事冒頭に戻って、「データの取得」の4つのセルを再実行してください。

まずインストール

```shell
!pip install faiss-gpu -U
import faiss
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting faiss-gpu
      Downloading faiss_gpu-1.7.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (85.5 MB)
    |████████████████████████████████| 85.5 MB 96 kB/s
    Installing collected packages: faiss-gpu
    Successfully installed faiss-gpu-1.7.2

総当たりのモデルの作成は一瞬です。

```python
%%time
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatIP(100)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add(normalized_dataset)
```

    CPU times: user 326 ms, sys: 359 ms, total: 684 ms
    Wall time: 745 ms

総当たりの時間は1.1秒です。CPUでは87秒かかっていましたから、80倍高速です。

```python
start = time.time()
faiss_brute_ans = gpu_index_flat.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_brute_ans[1], neighbors[:, :10])
```

    1.0996243953704834 sec
    1.0

IndexIVFFlatのモデルを作ってGPUに転送します。

```python
%%time
nlist = 100
quantizer = faiss.IndexFlatIP(100)
index = faiss.IndexIVFFlat(quantizer, 100, nlist, faiss.METRIC_INNER_PRODUCT)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
gpu_index.train(normalized_dataset)
gpu_index.add(normalized_dataset)
```

    CPU times: user 434 ms, sys: 53.5 ms, total: 488 ms
    Wall time: 429 ms

デフォルトで検索します。処理時間は0.2秒ですが、再現率が53%です。

```python
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    0.21199655532836914 sec
    0.53364

細かく検索範囲を広げて測定します。

```python
gpu_index.nprobe = 2
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    0.4109320640563965 sec
    0.67943

```python
gpu_index.nprobe = 3
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    0.6033580303192139 sec
    0.75168

```python
gpu_index.nprobe = 4
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    0.7939984798431396 sec
    0.79683

```python
gpu_index.nprobe = 5
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    0.9845168590545654 sec
    0.82667

```python
gpu_index.nprobe = 10
start = time.time()
faiss_searcher_ans = gpu_index.search(np.array(queries), 10)
print(time.time() - start, "sec")
compute_recall(faiss_searcher_ans[1], neighbors[:, :10])
```

    1.9291417598724365 sec
    0.90238

再現率90%で総当たりの処理時間を超えてしまいました。

GPUは確かに速いです。ただ、GPUのメモリに乗り切る規模であれば近似しないで総当たりするのが一番良いようです。
