function [accuracy] = CalAccuracy_MAE(test_outputs, test_target)
%Calculate the test accuracy for multi-class classification
[~, index1] = max(test_outputs, [], 2);
[~, index2] = max(test_target, [], 2);
t = abs(index1-index2)<=5;
accuracy = (sum(t))/(size(test_target, 1));
end