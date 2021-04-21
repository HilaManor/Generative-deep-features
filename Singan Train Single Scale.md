```python
z_opt = padded {Z*, 0, 0, 0, 0} (by noise) (only in first scale: noise changes every iter)
noise_ = padded noise (by noise) (changes every iter)

for every step in D
  GDS (train) by real image
  if first step & first epoch
      if first scale
          in_s = zeros
          prev = padded zeros (by image)
          z_prev = padded zeros (by noise)
          noise_amp = 1
      else
          prev = padded draw_concat 'rand' (by image)
              --> prev = previous generated image FROM RANDOM NOISE
          z_prev = padded draw_concat 'rec' (by image)
              --> z_prev = previous generate image FROM KNOWN NOISE
  else (not (first step and first epoch))
      prev = padded draw_concat 'rand' (by image)
          --> prev = previous generated image FROM (new) RANDOM NOISE


  if first scale:
      noise = noise_
  else:
      noise = noise_amp * noise_ + prev

  fake = G(noise, prev)
  GDS (train) by fake


for every step in G
  GDS (train) by fake
  if alpha != 0
      Z_opt = noise_amp * z_opt + z_prev
          --> in first scale: z_prev=0, z_opt random every iter
          --> else: z_prev = previous generate image FROM KNOWN NOISE
          -->         z_opt = 0 ({Z*,0,0,0,0,0})
      GDS (train) by reconstruction loss
          --> We want to ensure that there exists a specific
              set of input noise maps, which generates the original image x.
  else:
      Z_opt = z_opt (remember, only in first scale the noise changes every ITER)
          --> = 0 in all but first scale
      rec_loss = 0
```