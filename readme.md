# Vahadane论文normalization



### 简要说明

代码支持python3，未进行2、3兼容检查。

核心代码在`vahadane.py`。

需要额外安装的库有：spams, opencv。我spams安装的时候遇到一些问题，但是因为我的mac的原因。直接使用pip安装spams。opencv可自己找相应操作系统下的教程安装。

如果你装了jupyter notebook，可以运行一下main.ipynb，这里面是一个写好了的demo。

Report.md是我写给学姐看的，分析了一下参数选择的效果。可以不看。



### 使用方法

1 首先调用`utils.py`里的`read_image`函数读取图片。这个函数将输入图片从opencv默认的BGR转成RGB，并且对整体颜色进行了一些加强。读出来的图像矩阵是$n*m*3$的。

2 新建一个vahadane对象。两个LAMBDA已经调到最优。fast_mode是是否快速分解，getH_mode也会影响分解速度。ITER是getW时的一个参数。可以都试一下看看效果。

```python
vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
vhd.show_config()
```

3 矩阵分解

```python
Ws, Hs = vhd.stain_separate(source_image)
vhd.fast_mode=0;vhd.getH_mode=0;  # 在获取target的分解时，为了更精确，不用fast模式
Wt, Ht = vhd.stain_separate(target_image)
```

4 合成target

```python
img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
```

