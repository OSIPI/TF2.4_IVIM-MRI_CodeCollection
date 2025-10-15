#!/usr/bin/env Rscript

#Run like this:
#Rscript --vanilla tests/IVIMmodels/unit_tests/compare.r test_output.csv test_reference.csv reference_output.csv test_results.csv

# If this script fails:
# 1. Save the "Comparison" file from the run on Github, OR run this file directly
# 2. Find the file producted "test_reference.csv" on Github, or whatever the "reference_file" variable was called
# 3. This replaces "tests/IVIMmodels/unit_tests/reference_output.csv" in the repository
# 4. For the algorithm "IAR_LU_modified_mix" and "TCML_TechnionIIT_lsqtrf", replace the "f_f_alpha, Dp_f_alpha, D_f_alpha, f_t_alpha, Dp_t_alpha, D_t_alpha" columns with "0.01,0.01,0.01,0.0,0.0,0.0"

args = commandArgs(trailingOnly=TRUE)
# Define file paths
test_file <- "test_output.csv"
test_reference_file <- "test_reference.csv"
reference_file <- "" #"reference_output.csv"
test_result_file <- "test_results.csv"


if (length(args)>=1) {
   test_file = args[1]
}
if (length(args)>=2) {
    test_reference_file = args[2]
}
if (length(args)>=3) {
    reference_file = args[3]
}
if (length(args)>=4) {
    test_result_file = args[4]
}


# Load required libraries
library(tidyverse)
library(stats)
# library(testit)
library(assertr)

alpha <- 0.45  # be sensitive to changes

# Define desired columns to keep
keep_columns_reference <- c("Algorithm", "Region", "SNR", "f", "Dp", "D", "f_mu", "Dp_mu", "D_mu", "f_t_alpha", "Dp_t_alpha", "D_t_alpha", "f_f_alpha", "Dp_f_alpha", "D_f_alpha", "f_std", "Dp_std", "D_std", "f_df", "Dp_df", "D_df")
keep_columns_test <- c("Algorithm", "Region", "SNR", "index", "f", "Dp", "D", "f_fitted", "Dp_fitted", "D_fitted")

test <- read_csv(test_file) %>%
  select(all_of(keep_columns_test)) %>%
  # Convert Algorithm and Region to factors
  mutate(Algorithm = as.factor(Algorithm), Region = as.factor(Region))

# Group data by relevant factors
grouped_data <- test %>%
  group_by(Algorithm, Region, SNR, f, Dp, D)

# Combine data for easier comparison
# combined_data <- inner_join(reference, test, join_by(Algorithm, Region, SNR, f, Dp, D, index))

# Perform t-test for each value
summary_data <- grouped_data %>%
  summarize(
    # Calculate group means
    f_mu = mean(f_fitted),
    Dp_mu = mean(Dp_fitted),
    D_mu = mean(D_fitted),

    # Also insert default alpha values here
    f_t_alpha = alpha,
    Dp_t_alpha = alpha,
    D_t_alpha = alpha,
    f_f_alpha = alpha,
    Dp_f_alpha = alpha,
    D_f_alpha = alpha,

    # Calculate group standard deviations
    f_std = sd(f_fitted),
    Dp_std = sd(Dp_fitted),
    D_std = sd(D_fitted),

    # Degrees of freedom
    f_df = length(f_fitted) - 1,
    Dp_df = length(Dp_fitted) - 1,
    D_df = length(D_fitted) - 1,

    # Calculate group equivalence
    # f_fitted_equal = all(all.equal(f_fitted.x, f_fitted.y)),
    # Dp_fitted_equal = all(all.equal(Dp_fitted.x, Dp_fitted.y)),
    # D_fitted_equal = all(all.equal(D_fitted.x, D_fitted.y)),
    
    # Perform paired t-test for each value
    # f_fitted_p = t.test(f_fitted.x, f_fitted.y, paired = TRUE)$p.value,
    # Dp_fitted_p = t.test(Dp_fitted.x, Dp_fitted.y, paired = TRUE)$p.value,
    # D_fitted_p = t.test(D_fitted.x, D_fitted.y, paired = TRUE)$p.value
  )

# If no reference file, just report the test results and fail
write.csv(summary_data, test_reference_file, row.names=TRUE)

# Exit at this point if we don't have a reference file
if (nchar(reference_file) == 0) {
    stop("No reference file defined, stopping without testing.")
}


# Read data from CSV files and select only relevant columns
reference <- read_csv(reference_file) %>%
  select(all_of(keep_columns_reference)) %>%
  # Convert Algorithm and Region to factors
  mutate(Algorithm = as.factor(Algorithm), Region = as.factor(Region)) 

reference_combined <- inner_join(summary_data, reference, join_by(Algorithm, Region, SNR)) %>%
  group_by(Algorithm, Region, SNR)

# Run tests
test_results <- reference_combined %>%
  summarize(
    # f-tests
    f_ftest_lower = pf(f_std.x^2 / f_std.y^2, f_df.x, f_df.y, lower.tail=TRUE),
    f_ftest_upper = pf(f_std.x^2 / f_std.y^2, f_df.x, f_df.y, lower.tail=FALSE),
    Dp_ftest_lower = pf(Dp_std.x^2 / Dp_std.y^2, Dp_df.x, Dp_df.y, lower.tail=TRUE),
    Dp_ftest_upper = pf(Dp_std.x^2 / Dp_std.y^2, Dp_df.x, Dp_df.y, lower.tail=FALSE),
    D_ftest_lower = pf(D_std.x^2 / D_std.y^2, D_df.x, D_df.y, lower.tail=TRUE),
    D_ftest_upper = pf(D_std.x^2 / D_std.y^2, D_df.x, D_df.y, lower.tail=FALSE),

    # t-tests
    f_ttest_lower = pt((f_mu.x - f_mu.y) / (f_std.x / sqrt(f_df.x + 1)), df=f_df.y, lower.tail=TRUE),
    f_ttest_upper = pt((f_mu.x - f_mu.y) / (f_std.x / sqrt(f_df.x + 1)), df=f_df.y, lower.tail=FALSE),
    Dp_ttest_lower = pt((Dp_mu.x - Dp_mu.y) / (Dp_std.x / sqrt(Dp_df.x + 1)), df=Dp_df.y, lower.tail=TRUE),
    Dp_ttest_upper = pt((Dp_mu.x - Dp_mu.y) / (Dp_std.x / sqrt(Dp_df.x + 1)), df=Dp_df.y, lower.tail=FALSE),
    D_ttest_lower = pt((D_mu.x - D_mu.y) / (D_std.x / sqrt(D_df.x + 1)), df=D_df.y, lower.tail=TRUE),
    D_ttest_upper = pt((D_mu.x - D_mu.y) / (D_std.x / sqrt(D_df.x + 1)), df=D_df.y, lower.tail=FALSE),

    # alphas from reference
    f_f_alpha = f_f_alpha.y[1],
    Dp_f_alpha = Dp_f_alpha.y[1],
    D_f_alpha = D_f_alpha.y[1],
    f_t_alpha = f_t_alpha.y[1],
    Dp_t_alpha = Dp_t_alpha.y[1],
    D_t_alpha = D_t_alpha.y[1],
  )


test_results <- test_results %>%
  mutate(
    f_ftest_lower_null = f_ftest_lower >= f_f_alpha,
    f_ftest_upper_null = f_ftest_upper >= f_f_alpha,
    Dp_ftest_lower_null = Dp_ftest_lower >= Dp_f_alpha,
    Dp_ftest_upper_null = Dp_ftest_upper >= Dp_f_alpha,
    D_ftest_lower_null = D_ftest_lower >= D_f_alpha,
    D_ftest_upper_null = D_ftest_upper >= D_f_alpha,

    f_ttest_lower_null = f_ttest_lower >= f_t_alpha,
    f_ttest_upper_null = f_ttest_upper >= f_t_alpha,
    Dp_ttest_lower_null = Dp_ttest_lower >= Dp_t_alpha,
    Dp_ttest_upper_null = Dp_ttest_upper >= Dp_t_alpha,
    D_ttest_lower_null = D_ttest_lower >= D_t_alpha,
    D_ttest_upper_null = D_ttest_upper >= D_t_alpha,
  )


  # Write the t-test file
write.csv(test_results, test_result_file, row.names=TRUE)

# Fail if we had failures
test_results %>% verify(f_ftest_lower_null) %>%  summarize(n=n())
test_results %>% verify(f_ftest_upper_null) %>%  summarize(n=n())
test_results %>% verify(Dp_ftest_lower_null) %>%  summarize(n=n())
test_results %>% verify(Dp_ftest_upper_null) %>%  summarize(n=n())
test_results %>% verify(D_ftest_lower_null) %>%  summarize(n=n())
test_results %>% verify(D_ftest_upper_null) %>%  summarize(n=n())
test_results %>% verify(f_ttest_lower_null) %>%  summarize(n=n())
test_results %>% verify(f_ttest_upper_null) %>%  summarize(n=n())
test_results %>% verify(Dp_ttest_lower_null) %>%  summarize(n=n())
test_results %>% verify(Dp_ttest_upper_null) %>%  summarize(n=n())
test_results %>% verify(D_ttest_lower_null) %>%  summarize(n=n())
test_results %>% verify(D_ttest_upper_null) %>%  summarize(n=n())






# # Combine data for easier comparison
# reference_combined <- inner_join(grouped_data, reference, join_by(Algorithm, Region, SNR)) %>%
#   group_by(Algorithm, Region, SNR)

# # Run t-tests
# t_tests <- reference_combined %>%
#   summarize(
#     # Perform paired t-test for each value
#     f_fitted_p = t.test(f_fitted, mu = f_mu[1])$p.value,
#     Dp_fitted_p = t.test(Dp_fitted, mu = Dp_mu[1])$p.value,
#     D_fitted_p = t.test(D_fitted, mu = D_mu[1])$p.value
#   )

# # Extract p-values and assess significance, true is accept the null, false is reject
# t_tests <- t_tests %>%
#   mutate(
#     f_fitted_null = f_fitted_p >= alpha,
#     Dp_fitted_null = Dp_fitted_p >= alpha,
#     D_fitted_null = D_fitted_p >= alpha
#   )

# # Write the t-test file
# write.csv(t_tests, test_result_file, row.names=TRUE)

# # Fail if we had failures
# t_tests %>% verify(f_fitted_null)
# t_tests %>% verify(Dp_fitted_null)
# t_tests %>% verify(D_fitted_null)


# # Fail if we had failures (fallback)
# # failed_tests <- t_tests[!t_tests$f_fitted_null,]
# # print(failed_tests)
# # testit::assert(nrow(failed_tests) == 0)
# # failed_tests <- t_tests[!t_tests$Dp_fitted_null,]
# # print(failed_tests)
# # testit::assert(nrow(failed_tests) == 0)
# # failed_tests <- t_tests[!t_tests$D_fitted_null,]
# # print(failed_tests)
# # testit::assert(nrow(failed_tests) == 0)

# # TODO:
# # Could 
# # Could plot somehow?
# # Need to melt this data somehow to plot
# # grouped_plots <- grouped_data %>% do(plots=ggplot(data=.) + geom_boxplot(aes(f_fitted.x, f_fitted.y)))
