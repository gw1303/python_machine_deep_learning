# 연습문제 1

> plane CNN 모델 결과 시각화

``` python
accuracy = training_plane.history.history['acc']
val_accuracy = training_plane.history.history['val_acc']
loss = training_plane.history.history['loss']
val_loss = training_plane.history.history['val_loss']

plt.figure(figsize=(8,6))

plt.plot(accuracy, label='acc')
plt.plot(val_accuracy, label='val_acc')
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')

plt.xlabel('epoch')
plt.legend()
```