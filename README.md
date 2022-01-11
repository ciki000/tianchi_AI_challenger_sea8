# 初赛

## 记录

1. PGD12d+2PGD12r+3PGD12d+light+wasserstein: 98.0788
2. PGD12d+2PGD12r+3PGD12d+gaussian+wasserstein: 97.5600
3. PGD12d+2PGD12r+3PGD12d+3PGD12r+wasserstein: 97.5082
4. PGD12d+2PGD12r+3PGD12d+salt+wasserstein: 97.3802
5. PGD12d+2PGD12r+3PGD12r+light+wasserstein: 98.2233
6. PGD12r+2PGD12r+3PGD12r+light+wasserstein: 97.2687
7. PGD12d+2PGD12r+3PGD12r+light+wasserstein_init0.22: 98.0775
8. PGD12d+2PGD12r+3PGD12r+light_85-95+wasserstein: 98.1613
9. PGD12d+2PGD12r+3PGD12r+light_PGD2r+wasserstein: 97.7342
10. PGD12d+2PGD12r+3PGD12r+light_78-85+wasserstein: 98.0922
11. PGD12d+2PGD12r+3PGD12r+light+wasserstein+softlabel: 97.9325
12. PGD12d+2PGD12r+test+light+wasserstein: 97.8097

## 心得

densenet121比resnet50鲁棒

numpy数组转化为uint8类型前记得round

# 复赛
## 记录
<table>
   <tr>
      <td></td>
      <td>test</td>
      <td>test_light</td>
      <td>test_blur</td>
      <td>test_PGD12_r</td>
      <td>test_PGD12_d</td>
      <td>test_wasserstein</td>
   </tr>
   <tr>
      <td>train_base</td>
      <td>93.05/93.74</td>
      <td>83.45/86.58</td>
      <td>53.91/58.64</td>
      <td>35.68/36.3</td>
      <td>21.98/13.07</td>
      <td>86.41/88.27</td>
   </tr>
   <tr>
      <td>train_PGD12-d</td>
      <td>37.44/44.83</td>
      <td>13.78/16.31</td>
      <td>13.99/18.07</td>
      <td>66.74/67.69</td>
      <td>80.85/81.12</td>
      <td>18.22/25.69</td>
   </tr>
   <tr>
      <td>train_PGD8-d</td>
      <td>49.67/53.26</td>
      <td>19.9/20.0</td>
      <td>24.78/20.2</td>
      <td>63.3/69.52</td>
      <td>82.23/82.28</td>
      <td>31.22/33.18</td>
   </tr>
   <tr>
      <td>train(_ P8d P8r l _)</td>
      <td>90.02/91.8</td>
      <td>89.52/91.41</td>
      <td>75.58/73.32</td>
      <td>74.05/76.26</td>
      <td>78.23/82.65</td>
      <td>82.82/84.45</td>
   </tr>
   <tr>
      <td>train(_ P8d P8r _ _)</td>
      <td>92.19/92.11</td>
      <td>79.06/82.48</td>
      <td>77.79/68.19</td>
      <td>77.36/78.67</td>
      <td>81.08/84.6</td>
      <td>85.74/83.69</td>
   </tr>
   <tr>
      <td>train(_ P8d P8r P4d P4r)</td>
      <td>91.64/90.19</td>
      <td>75.96/79.02</td>
      <td>77.6/67.75</td>
      <td>80.26/74.12</td>
      <td>84.95/83.25</td>
      <td>85.34/80.29</td>
   </tr>
</table>