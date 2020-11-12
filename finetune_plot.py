import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


steps = [i* 4000 for i in range(1, 11)]
resCIDER_noLabel = [0.9167959223426917, 
                 1.0237224443519906, 
                 1.0363495280177946, 
                 1.0659786128213262, 
                 1.0848930718643104, 
                 1.1100207325786775, 
                 1.1019678298104678,
                 1.1004035438046056,
                 1.0987989587607652,
                 1.1225917796038314]
resCIDER_noImg = [0.9032794222,
                  0.9723122226,
                  0.9732555155,
                  0.9702411857,
                  0.9753644601578392,
                  0.981117484661295,
                  0.9857072192726355,
                  0.973542006490171,
                  0.9773295586842116,
                  0.9787185070027324]

resCIDER_noimg = [0.5*i for i in range(1, 11)]
df=pd.DataFrame({'steps': steps, 'finetune_noObjectTag': resCIDER_noLabel, 'finetune_noImgFeat': resCIDER_noImg})#'noImgFeature': resCIDER_noimg})
plt.plot( 'steps', 'finetune_noObjectTag', data=df, marker='o', color='olive', linewidth=2)
plt.plot( 'steps', 'finetune_noImgFeat', data=df, marker='o', color='skyblue', linewidth=2)
plt.title('Finetune')
plt.xlabel('steps')
plt.ylabel('CIDEr')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('finetune.png')
