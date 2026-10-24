args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4 || length(args) > 5) {
  stop("Usage: prediction_bridge.R <model_rds> <model_key> <input_csv> <output_csv> [explanation_csv]")
}

model_path <- args[[1]]
model_key <- args[[2]]
input_path <- args[[3]]
output_path <- args[[4]]
explanation_path <- if (length(args) >= 5) args[[5]] else NULL

suppressPackageStartupMessages({
  library(tidymodels)
  library(xgboost)
  library(ranger)
  library(rpart)
  library(nnet)
})

models <- readRDS(model_path)
if (!(model_key %in% names(models))) {
  stop(paste("Model key not found:", model_key))
}

new_data <- read.csv(
  input_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA")
)

# Target severity_class must never be included in inference input
if ("severity_class" %in% names(new_data)) {
  new_data$severity_class <- NULL
}

if (nrow(new_data) != 1) {
  stop("Prediction input must contain exactly one row.")
}

fitted_model <- models[[model_key]]
class_prediction <- predict(fitted_model, new_data = new_data, type = "class")
prob_prediction <- predict(fitted_model, new_data = new_data, type = "prob")

pred_class_val <- as.character(class_prediction$.pred_class)

result <- data.frame(
  predicted_class = pred_class_val,
  stringsAsFactors = FALSE
)

for (column_name in names(prob_prediction)) {
  result[[column_name]] <- as.numeric(prob_prediction[[column_name]])
}

write.csv(result, output_path, row.names = FALSE, na = "")

# Compute local prediction explanation if requested
if (!is.null(explanation_path) && nchar(explanation_path) > 0) {
  explain_prediction <- function(model_obj, key, data_row, pred_class, probs) {
    tryCatch({
      if (key %in% c("xgboost", "gradient_boosting")) {
        rec <- extract_recipe(model_obj)
        booster <- extract_fit_engine(model_obj)
        baked <- bake(rec, new_data = data_row)
        contrib <- predict(booster, newdata = as.matrix(baked), predcontrib = TRUE)
        
        classes <- c("Low", "Medium", "High")
        pred_idx <- match(pred_class, classes)
        if (is.na(pred_idx)) pred_idx <- 1
        
        shap_vals <- contrib[1, pred_idx, ]
        bias <- as.numeric(shap_vals["(Intercept)"])
        feat_vals <- shap_vals[names(shap_vals) != "(Intercept)"]
        
        df_exp <- data.frame(
          feature = names(feat_vals),
          contribution = as.numeric(feat_vals),
          stringsAsFactors = FALSE
        )
        df_exp <- df_exp[df_exp$contribution != 0, ]
        df_exp$abs_val <- abs(df_exp$contribution)
        df_exp <- df_exp[order(-df_exp$abs_val), ]
        df_exp$abs_val <- NULL
        
        df_exp$direction <- ifelse(
          df_exp$contribution > 0,
          paste("Increases", pred_class, "prediction confidence"),
          paste("Decreases", pred_class, "prediction confidence")
        )
        df_exp$method <- "TreeSHAP (Exact Shapley Contributions)"
        df_exp$pred_class <- pred_class
        df_exp$baseline_bias <- bias
        return(head(df_exp, 25))
        
      } else if (key == "decision_tree") {
        rec <- extract_recipe(model_obj)
        rp <- extract_fit_engine(model_obj)
        baked <- bake(rec, new_data = data_row)
        
        splits_df <- rp$frame
        splits_df <- splits_df[splits_df$var != "<leaf>", ]
        split_vars <- unique(as.character(splits_df$var))
        
        exp_rows <- list()
        for (v in split_vars) {
          if (v %in% names(baked)) {
            val <- as.numeric(baked[[v]][1])
            exp_rows[[length(exp_rows) + 1]] <- data.frame(
              feature = v,
              contribution = round(val, 4),
              direction = paste0("Decision branch threshold variable (standardized value: ", round(val, 2), ")"),
              method = "Decision Tree Split Path",
              pred_class = pred_class,
              baseline_bias = 0,
              stringsAsFactors = FALSE
            )
          }
        }
        if (length(exp_rows) > 0) {
          return(do.call(rbind, exp_rows))
        }
        
      } else {
        # Local perturbation sensitivity for random_forest, extra_trees, mlp
        pred_prob_col <- paste0(".pred_", pred_class)
        base_p <- as.numeric(probs[[pred_prob_col]])
        
        candidate_features <- c(
          "has_disruption", "is_disruption", "is_data_theft", "is_hijacking",
          "is_ransomware", "is_doxing", "target_critical_infrastructure",
          "target_government", "target_corporate", "target_media",
          "target_education", "target_social_groups", "attribution_available",
          "initiator_attributed", "state_affiliated_initiator", "phishing_used",
          "exploit_public_facing", "data_exfiltration", "mitre_data_encryption"
        )
        candidate_features <- candidate_features[candidate_features %in% names(data_row)]
        
        pert_list <- list()
        for (feat in candidate_features) {
          orig_val <- data_row[[feat]][1]
          if (!is.na(orig_val)) {
            pert_data <- data_row
            pert_data[[feat]][1] <- ifelse(orig_val == 1, 0, 1)
            pert_prob <- predict(model_obj, new_data = pert_data, type = "prob")
            new_p <- as.numeric(pert_prob[[pred_prob_col]])
            delta_p <- base_p - new_p
            
            if (abs(delta_p) > 1e-4) {
              pert_list[[length(pert_list) + 1]] <- data.frame(
                feature = feat,
                contribution = delta_p,
                direction = ifelse(
                  delta_p > 0,
                  paste0("Active value (", orig_val, ") increases ", pred_class, " probability (+", round(abs(delta_p)*100, 1), "%)"),
                  paste0("Active value (", orig_val, ") decreases ", pred_class, " probability (-", round(abs(delta_p)*100, 1), "%)")
                ),
                method = "Local Sensitivity (Marginal Perturbation)",
                pred_class = pred_class,
                baseline_bias = base_p,
                stringsAsFactors = FALSE
              )
            }
          }
        }
        if (length(pert_list) > 0) {
          df_pert <- do.call(rbind, pert_list)
          df_pert <- df_pert[order(-abs(df_pert$contribution)), ]
          return(head(df_pert, 20))
        }
      }
    }, error = function(e) {
      return(data.frame(
        feature = "Diagnostic Note",
        contribution = 0,
        direction = paste("Local explanation calculation:", conditionMessage(e)),
        method = "Diagnostic Notice",
        pred_class = pred_class,
        baseline_bias = 0,
        stringsAsFactors = FALSE
      ))
    })
    
    return(data.frame(
      feature = "Baseline Profile",
      contribution = 0,
      direction = "Input combination aligns with standard baseline profile for this severity class",
      method = "Baseline Alignment",
      pred_class = pred_class,
      baseline_bias = 0,
      stringsAsFactors = FALSE
    ))
  }
  
  exp_result <- explain_prediction(fitted_model, model_key, new_data, pred_class_val, prob_prediction)
  write.csv(exp_result, explanation_path, row.names = FALSE, na = "")
}
