import pandas as pd
import matplotlib.pyplot as plt

modelName = 'psnNet_a-0.01_e-100_n-1000_b-64_a-relu_p-200_t-0_l-i3232364c364c3128m20c3256m20c3512c31024t3512t3256b2t3128b2t364t31'

# Read log file
history = pd.read_csv(modelName + '.log', sep = ',', engine='python')

def plot_history(epochStart, epochEnd, titles=['MSE loss', 'MAE'], plotLr=False,
                 args=[['loss', 'val_loss'], ['mae', 'val_mae']],
                 nRows=1, nColumns=2):
  hist = history[epochStart:epochEnd]
  fig = plt.figure(figsize=(8*nColumns, 4*nRows), dpi=120)
  fig.subplots_adjust(hspace=.5)

  for i, names in enumerate(args):
    ax = fig.add_subplot(nRows, nColumns, i+1)
    ax.set_yscale('log')
    for n in names:
      assert n in names
      plt.plot(hist[n])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    ax.title.set_text(titles[i])
    legend = args[i]
    if plotLr:
      plt.plot(hist['lr'], color='C3')
      legend.append('lr')
    plt.legend(legend, loc='upper right')

  fig.savefig('{}_h_{}.png'.format(modelName, epochStart, epochEnd), bbox_inches='tight')
  plt.close(fig)

# Plot history in multiple steps
e = 100
plot_history(0, e)
#plot_history(e, 800)

