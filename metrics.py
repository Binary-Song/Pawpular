import sklearn.metrics as met
import numpy as np
 
class AccuracyScore:

    def __init__(self, take_prob = True):
        '''
        take_prob (bool): True:输入各类别概率, False:输入最可能类别的标签。 
        '''        
        self.name = 'accuracy'
        self.take_prob = take_prob
    
    def __call__(self, targets, outputs):
        '''
        targets (numpy, N*1): 目标列，此为正确的标签
        outputs (numpy, N*1 or N*classes): 输出列，当take_prob为假时，形状为N*1，否则为N*classes。
        '''
        if self.take_prob == False:
            return met.accuracy_score(targets, outputs)
        else:
            output_labels = np.argmax(outputs, axis=1)
            return met.accuracy_score(targets, output_labels)

class F1Score:
    
    def __init__(self, average = None, take_prob = True):
        '''
        average (str): 平均方式，见f1_score。
        take_prob (bool): 接受各类别概率还是接受最可能类别的标签。
        '''
        self.average = average
        self.name = 'f1'
        self.take_prob = take_prob

        if average is not None:
            self.name += f'-{average}'
    
    def __call__(self, targets, outputs):
        '''
        targets (numpy, N*1): 目标列，此为正确的标签
        outputs (numpy, N*1 or N*classes): 输出列，当take_prob为假时，形状为N*1，否则为N*classes。
        '''
        if self.take_prob == False:
            return met.f1_score(targets, outputs, average=self.average)
        else:
            output_labels = np.argmax(outputs, axis=1)
            return met.f1_score(targets, output_labels, average=self.average)

class RocAucScore:
    
    def __init__(self, average = None, multi_class = 'raise'):
        '''
        average (str): 平均方式，见roc_auc_score。
        multi_class (str): 多分类方式，见roc_auc_score。 
        '''
        self.average = average
        self.multi_class = multi_class

        self.name = 'rocauc'
        if average is not None:
            self.name += f'-{average}'
    
    def __call__(self, targets, outputs):
        '''
        targets (numpy, N*1): 目标列，此为正确的标签.
        outputs (numpy, N*classes): 输出列，此为各类别的概率（和为1）.
        '''
        return met.roc_auc_score(targets, outputs, average=self.average, multi_class=self.multi_class)
    
class R2Score:

    def __init__(self ):
        ''' 
        ''' 
        self.name = 'r-square' 
    
    def __call__(self, targets, outputs):
        '''
        targets (numpy, N*1): 目标列 
        outputs (numpy, N*1): 输出列 
        '''
        return met.r2_score(targets, outputs)

class NegMeanAbsError:

    def __init__(self ):
        ''' 
        ''' 
        self.name = 'neg-mean-abs-err' 
    
    def __call__(self, targets, outputs):
        '''
        targets (numpy, N*1): 目标列 
        outputs (numpy, N*1): 输出列 
        '''
        return -met.mean_absolute_error(targets, outputs)
