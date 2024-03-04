function arrFeatures = returnFeatures(param_arr)

measure_set = {'Mean','Median','StDev','Skewness','Avg_Signal'};

arrFeatures = zeros(1,5);

arrFeatures(1) = mean(param_arr);
arrFeatures(2) = median(param_arr);
arrFeatures(3) = std(param_arr);
arrFeatures(4) = skewness(param_arr);