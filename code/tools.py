import numpy as np
import subprocess

# eval
def rmsle(true, pred):
    '''
        evaluation metric for the competition

    Args:
        true : ground tru values
        pred : model predictions 

    Returns:
        root mean squared log error
    '''
    return np.sqrt(np.mean((np.log1p(true) - np.log1p(pred))**2))


def evaluate(model, x_trn, y_trn, x_val, y_val, fn):
    '''
        evaluate trn, val loss / R2 score / OOB score

    
    '''
    trn_pred = model.predict(x_trn)
    val_pred = model.predict(x_val)
    
    print('# trn loss : {}'.format(fn(y_trn, trn_pred)))
    print('# val loss : {}'.format(fn(y_val, val_pred)))
    
    print('# trn R2 : {}'.format(model.score(x_trn, y_trn)))
    print('# val R2 : {}'.format(model.score(x_val, y_val)))

    if hasattr(model, 'oob_score_') and model.oob_score:
        print('# OOB_score : {}'.format(model.oob_score_))

# submit
def submit_kaggle_result(competition, fname, msg):
    '''
        submit kaggle competition result with message
    '''
    submit_cmd = 'kaggle competitions submit -c {} -f {} -m "{}"'.format(competition, fname, msg)
    subprocess.call(submit_cmd, stderr=subprocess.STDOUT, shell=True)


def check_kaggle_result(competition, fname=None):
    '''
        check kaggle competition result, marking BEST result and THIS result
    '''
    check_cmd = 'kaggle competitions submissions -c {}'.format(competition)
    result = subprocess.check_output(check_cmd, stderr=subprocess.STDOUT, shell=True).split('\n')

    best_lb = 9999
    for line in result:
        if line == '':
            break
        try:
            lb_score = float(line.split()[-1])
            if lb_score < best_lb:
                best_lb = lb_score
        except:
            pass

    print('')
    for line in result:
        if 'Warning' in line:
            continue
        if fname is not None and fname in line:
            line += '(THIS) '
        if str(best_lb) in line:
            line += '(BEST) '
        print(line)

# permutation importance
