# compare model results directories with_movements and without_movements, to truth

import os
import shutil

truth = 'Carol_truth'
truth_with_movements = os.listdir(f"{truth}/with_movements")
truth_without_movements = os.listdir(f"{truth}/without_movements")
pred_with_movements = os.listdir("results/with_movements")
pred_without_movements = os.listdir("results/without_movements")

TP, TN, FP, FN = [0] * 4

for file in pred_with_movements:
    if file in truth_with_movements:
        TP += 1
    elif file in truth_without_movements:
        FP += 1
        shutil.copy(f"{truth}/without_movements/{file}", f"wronged/without_spadefoot/{file}")
    else:
        print('not found in truth:  ', file)

for file in pred_without_movements:
    if file in truth_with_movements:
        FN += 1
        shutil.copy(f"{truth}/with_movements/{file}", f"wronged/with_spadefoot/{file}")
    elif file in truth_without_movements:
        TN += 1
    else:
        print('not found in truth:  ', file)

print('total images:', len(pred_with_movements) + len(pred_without_movements))
print(f"true positive: {TP}   true negative: {TN}   false positive: {FP}   false negative: {FN}")
print(f"precision: {TP/(TP+FP)}   recall: {TP/(TP+FN)}")