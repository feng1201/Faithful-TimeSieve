import pywt
import numpy as np

import torch
import pywt
import numpy as np
import torch
import pywt
import numpy as np

def wavelet_decomposition_and_reconstruction(data, wavelet='db1', mode='symmetric'):
    """
    对Tensor数据进行小波变换，分解为趋势和细节部分，然后重构原始信号。

    参数:
    - data: 输入数据，形状为(time_steps, num_features)，为Tensor类型。
    - wavelet: 使用的小波类型，默认为'Daubechies 1'。
    - mode: 小波变换的填充模式，默认为'symmetric'。

    返回:
    - reconstructed_data: 小波逆变换重构后的数据，形状与输入相同，为Tensor类型。
    """
    data_np = data.numpy()  # 将Tensor转换为NumPy数组
    num_features = data_np.shape[1]
    reconstructed_data_np = np.zeros_like(data_np)

    for feature in range(num_features):
        signal = data_np[:, feature]
        # 执行一级小波分解，得到趋势和细节部分
        cA, cD = pywt.dwt(signal, wavelet, mode=mode)
        # 使用趋势和细节部分重构信号
        reconstructed_signal = pywt.idwt(cA, cD, wavelet, mode=mode)
        # 由于重构可能会引入小的尺寸变化，我们取原始信号长度的数据
        reconstructed_data_np[:, feature] = reconstructed_signal[:data_np.shape[0]]

    # 将结果转换回Tensor
    reconstructed_data = torch.tensor(reconstructed_data_np, dtype=data.dtype)
    return reconstructed_data

# 示例使用
data_tensor = torch.rand(96, 7)  # 示例数据，实际使用时替换为你的数据
reconstructed_data_tensor = wavelet_decomposition_and_reconstruction(data_tensor)


