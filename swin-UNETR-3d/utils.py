def window_partition3d(x,window_size):
  """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size (window_d, window_h, window_w)

    Returns:
        windows: (num_windows*B, window_d, window_h, window_w, C)
  """
  B,D,H,W,C=x.shape
  x=x.view(B,
      D//window_size[0],window_size[0],
      H//window_size[1],window_size[1],
      W//window_size[2],window_size[2],
      C
  )
  windows=x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1,window_size[0],window_size[1],window_size[2],C)
  return windows

def window_reverse3d(windows,window_size,D,H,W):
  """
    Args:
        windows: (num_windows*B, window_d, window_h, window_w, C)
        window_size (tuple[int]): Window size
        D (int): Depth of volume
        H (int): Height of volume
        W (int): Width of volume

    Returns:
        x: (B, D, H, W, C)
  """
  num_windows_per_batch = (D // window_size[0]) * (H // window_size[1]) * (W // window_size[2])
  B = windows.shape[0] // num_windows_per_batch
  x=windows.view(B,
                D//window_size[0],H//window_size[1],W//window_size[2],
                window_size[0],window_size[1],window_size[2],
                -1)
  x=x.permute(0,1,4,2,5,3,6,7).contiguous().view(B,D,H,W,-1)
  return x