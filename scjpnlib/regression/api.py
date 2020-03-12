import pandas as pd
from IPython.core.display import HTML
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.graphics.utils as smgrpu
from matplotlib.gridspec import GridSpec
import statsmodels.tools.tools as smtt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.graphics.regressionplots as smgrp
from ..utils import *
from itertools import combinations
from sklearn.model_selection import cross_val_score

plot_edge = 4

def histograms(df):
    feats = df.columns
    
    s_html = "<h3>Distributions:</h3><ul>"
    s_html += "<li><b>feature set</b>: {}</li>".format(feats)
    s_html += "</ul>"
    display(HTML(s_html))
    
    n_feats = len(feats)
    
    r_w = 4*plot_edge if n_feats > plot_edge else (n_feats*4 if n_feats > 1 else plot_edge)
    r_h = plot_edge if n_feats > 4 else (plot_edge if n_feats > 1 else plot_edge)
    
    c_n = 4 if n_feats > 4 else n_feats
    r_n = n_feats/c_n
    r_n = int(r_n) + (1 if n_feats > 4 and r_n % int(r_n) != 0 else 0)        

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)
    unused = list(range(0, len(axes.flatten()))) if n_feats > 1 else [0]

    included_feats = []
    for index, feat in enumerate(feats):
        ax = fig.add_subplot(r_n, c_n, index+1)
        sns.distplot(df[feat], ax=ax)
        plt.xlabel(feat)
        included_feats.append(feat)
        unused.remove(index)

    flattened_axes = axes.flatten() if n_feats > 1 else [axes]
    for u in unused:
        fig.delaxes(flattened_axes[u])

    fig.tight_layout()
    plt.show();

    return included_feats

def histograms_comparison(df1, df2, title1, title2):    
    s_html = "<h3>Distribution Comparisons:</h3><ul>"
    s_html += "</ul>"
    display(HTML(s_html))
    
    n_feats = len(df1.columns)
    
    r_w = 2*plot_edge
    r_h = plot_edge
    
    c_n = 2
    r_n = n_feats       

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)

    for index in range(0, n_feats):
        ax1 = fig.add_subplot(r_n, c_n, 2*index+1)
        sns.distplot(df1.iloc[:, index], ax=ax1)
        ax1.set_xlabel(df1.columns[index])
        ax1.set_title(title1)

        ax2 = fig.add_subplot(r_n, c_n, 2*index+2)
        sns.distplot(df2.iloc[:, index], ax=ax2)
        ax2.set_xlabel(df2.columns[index])
        ax2.set_title(title2)

    fig.tight_layout()
    plt.show();

def plot_corr(df, filter=None):
    corr = df[filter].corr() if (filter is not None and len(filter) > 0) else df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(22, 18))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr
        , mask=mask
        , cmap=cmap
        , vmax=1.0
        , center=0
        , square=True
        , linewidths=.5
        , cbar_kws={"shrink": .5}
    )
    plt.title("Feature Correlations")
    plt.show();

def feature_regression_summary(
    df
    , feat_idx
    , target
    , model_fit_results
    , display_regress_diagnostics=False):

    feat = df.columns[feat_idx]
    
    v = vif(np.matrix(df), feat_idx)
    colinear = v > 10
    
    if display_regress_diagnostics:   
        # ‘endog versus exog’, ‘residuals versus exog’, ‘fitted versus exog’ and ‘fitted plus residual versus exog’
        fig = plt.figure(constrained_layout=True, figsize=(2.25*plot_edge,4*plot_edge))
        fig = smgrpu.create_mpl_fig(fig)
        gs = GridSpec(3, 2, figure=fig)
        ax_fit = fig.add_subplot(gs[0, 0])
        ax_partial_residuals = fig.add_subplot(gs[0, 1])
        ax_partregress = fig.add_subplot(gs[1, 0])
        ax_ccpr = fig.add_subplot(gs[1, 1])
        ax_dist = fig.add_subplot(gs[2:4, 0:2])
        
        exog_name, exog_idx = smgrpu.maybe_name_or_idx(feat, model_fit_results.model)
        smresults = smtt.maybe_unwrap_results(model_fit_results)
        y_name = smresults.model.endog_names
        x1 = smresults.model.exog[:, exog_idx]
        prstd, iv_l, iv_u = wls_prediction_std(smresults)
        
        # endog versus exog
        # use wrapper since it's availab;e!
        sm.graphics.plot_fit(model_fit_results, feat, ax=ax_fit)
        
        # residuals versus exog
        ax_partial_residuals.plot(x1, smresults.resid, 'o')
        ax_partial_residuals.axhline(y=0, color='black')
        ax_partial_residuals.set_title('Residuals versus %s' % exog_name, fontsize='large')
        ax_partial_residuals.set_xlabel(exog_name)
        ax_partial_residuals.set_ylabel("resid")
        
        # Partial Regression plot: fitted versus exog
        exog_noti = np.ones(smresults.model.exog.shape[1], bool)
        exog_noti[exog_idx] = False
        exog_others = smresults.model.exog[:, exog_noti]
        from pandas import Series
        smgrp.plot_partregress(
            smresults.model.data.orig_endog
            , Series(
                x1
                , name=exog_name
                , index=smresults.model.data.row_labels
            )
            , exog_others
            , obs_labels=False
            , ax=ax_partregress
        )        
        
        # CCPR: fitted plus residual versus exog
        # use wrapper since it's availab;e!
        sm.graphics.plot_ccpr(model_fit_results, feat, ax=ax_ccpr)
        ax_ccpr.set_title('CCPR Plot', fontsize='large')

        sns.distplot(df[feat], ax=ax_dist)
        
        fig.suptitle('Regression Plots for %s' % exog_name, fontsize="large")
        #fig.tight_layout()
        #fig.subplots_adjust(top=.90)        
        plt.show()  
        
        display(
            HTML(
                "Variance Inflation Factor (<i>VIF</i>) for <b>{}</b>: <b>{}</b> {}".format(
                    feat
                    , round(v, 2)
                    , "$\\le 10 \\iff$ low colinearity" if not colinear else "$> 10 \\iff$ <b>HIGH COLINEARITY</b>"
                )
            )
        ) 
        display(
            HTML(
                "<b><i>p-value</i></b> (<i>VIF</i>) for <b>{}</b>: <b>{}</b><br><br>".format(
                    feat
                    , model_fit_results.pvalues[feat_idx+1]
                )
            )
        )
    
    return v

#this method simply houses the combinations (n-choose-k) used in cross-validation
def cv_build_feature_combinations(X, reverse=False, upper_bound=2**18, boundary_test=False):
    feat_combos = dict()
    
    r = range(len(X.columns), 0, -1) if reverse else range(1, len(X.columns)+1)  # build up from potentially worst case
        
    n = max(r)
    
    # determine whether or not we will exhause memory!
    len_total_combos = 0
    
    s_n_choose_k = "{} \\choose {}"
    if not boundary_test:
        s_total_combos = "<br><br>Builing all $\\sum_{i=" + str(min(r)) + "}^{" + str(n) + "}{" + s_n_choose_k.format(n, "i") + "}=$"
        for k in r:
            s_total_combos += " ${" + s_n_choose_k.format(n, k) + "}$" + (" +" if k!=n else "")
            len_combos = nCr(n, k)
            len_total_combos += len_combos
        s_total_combos += " $={}$ combinations of feature set: {}...".format(len_total_combos, X.columns)
        display(HTML(s_total_combos))
    else:
        len_total_combos = 2**n - 1
    
    if len_total_combos > upper_bound:
        display(HTML("<h2><font color=\"red\">Building all combinations of {} features would result in cross-validating {} models which will most likely exhaust memory (exceeds upper bound: {})!  Please reduce the size of the feature set and try again!</font></h2>".format(n, len_total_combos, upper_bound)))
        feat_combos = None
    else:
        if not boundary_test:
            for k in r:
                feat_combos_of_length_k = list(combinations(X, k))
                feat_combos[k] = feat_combos_of_length_k
        else:
            display(HTML("<h2><font color=\"green\">Boundary test PASSED!  Building all combinations of {} features will result in cross-validating {} models, which is less than upper bound: {}.</font></h2>".format(n, len_total_combos, upper_bound)))
    
    display(HTML("All done!"))    
    
    return (feat_combos, len_total_combos)

#below are the list of constant string-literals used for cross-validation
mse = 'mse'
mse_train = mse + '_train'
mse_test = mse + '_test'
delta_mse = 'delta_' + mse
mse_train_and_mse_test = mse_train + '__and__' + mse_test
mse_and_delta_mse = mse + '__and__' + delta_mse
rmse = 'rmse'
rmse_train = rmse + '_train'
rmse_test = rmse + '_test'
delta_rmse = 'delta_' + rmse
rmse_train_and_rmse_test = rmse_train + '__and__' + rmse_test
rmse_and_delta_rmse = rmse + '__and__' + delta_rmse
rsquared = 'rsquared'
adjusted_rsquared = 'rsquared_adj'
condition_no = 'condition_no'
pvals = 'pvals'
condition_no_and_adjusted_rsquared = condition_no + '__and__' + adjusted_rsquared
condition_no_and_rmse_and_delta_rmse = condition_no + '__and__' + rmse_and_delta_rmse
condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse = \
    condition_no \
    + '__and__' + pvals \
    + '__and__' + rsquared \
    + '__and__' + adjusted_rsquared \
    + '__and__' + rmse_and_delta_rmse

cv_scoring_methods = [
    mse_train_and_mse_test
    , mse_and_delta_mse
    , rmse_train_and_rmse_test
    , rmse_and_delta_rmse
    , adjusted_rsquared
    , condition_no
    , condition_no_and_adjusted_rsquared
    , condition_no_and_rmse_and_delta_rmse
    , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
]

#this method does the scoring for cross-validation (using k-folds)
def cv_score(
    X
    , y
    , feat_combo
    , folds=5
    , scoring_method=rmse_train_and_rmse_test):

    if scoring_method not in cv_scoring_methods:
        raise ValueError("Unknown scoring_method: '{}'".format(scoring_method))

    if scoring_method == mse_train_and_mse_test:
        scores_df = pd.DataFrame(columns=[mse_train, mse_test])

    elif scoring_method == mse_and_delta_mse:
        scores_df = pd.DataFrame(columns=[mse, delta_mse])

    elif scoring_method == rmse_train_and_rmse_test:
        scores_df = pd.DataFrame(columns=[rmse_train, rmse_test])

    elif scoring_method == rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[rmse, delta_rmse])

    elif scoring_method == adjusted_rsquared:
        scores_df = pd.DataFrame(columns=[adjusted_rsquared])

    elif scoring_method == condition_no:
        scores_df = pd.DataFrame(columns=[condition_no])

    elif scoring_method == condition_no_and_adjusted_rsquared:
        scores_df = pd.DataFrame(columns=[condition_no, adjusted_rsquared])

    elif scoring_method == condition_no_and_rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[condition_no, rmse, delta_rmse])

    elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[condition_no, rsquared, adjusted_rsquared, pvals, rmse, delta_rmse])


    f = y.columns[0] + '~' + "+".join(feat_combo)
    scores = []

    if folds > 1:
        train_test_indices = KFold(n_splits=folds).split(X)
    else:
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
        train_test_indices = [(X_train.index, X_test.index)]

    for train_index, test_index in train_test_indices:
        X_train, X_test = X.iloc[train_index][feat_combo], X.iloc[test_index][feat_combo]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        data_fin_df = pd.concat([y_train, X_train], axis=1, join='inner').reset_index()
        model_fit_results = ols(formula=f, data=data_fin_df).fit()

        if scoring_method == mse_train_and_mse_test \
            or scoring_method == mse_and_delta_mse \
            or scoring_method == rmse_train_and_rmse_test \
            or scoring_method == rmse_and_delta_rmse \
            or scoring_method == condition_no_and_rmse_and_delta_rmse \
            or scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:

            y_hat_train = model_fit_results.predict(X_train)
            y_hat_test = model_fit_results.predict(X_test)
            _mse_train = mean_squared_error(y_train, y_hat_train)
            _mse_test = mean_squared_error(y_test, y_hat_test)

            if scoring_method == mse_train_and_mse_test:
                data = [
                    {
                        mse_train: _mse_train
                        , mse_test: _mse_test
                    }
                ]

            elif scoring_method == rmse_train_and_rmse_test:
                data = [
                    {
                        rmse_train: np.sqrt(_mse_train)
                        , rmse_test: np.sqrt(_mse_test)
                    }
                ]

            elif scoring_method == mse_and_delta_mse:
                data = [
                    {
                        mse: _mse_train
                        , delta_mse: abs(_mse_test - _mse_train)
                    }
                ]

            elif scoring_method == rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            elif scoring_method == condition_no_and_rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        condition_no: model_fit_results.condition_number
                        , rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        condition_no: model_fit_results.condition_number
                        , rsquared: model_fit_results.rsquared
                        , adjusted_rsquared: model_fit_results.rsquared_adj
                        , pvals: model_fit_results.pvalues
                        , rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == adjusted_rsquared:
            data = [{adjusted_rsquared: model_fit_results.rsquared_adj}]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == condition_no:
            data = [{condition_no: model_fit_results.condition_number}]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == condition_no_and_adjusted_rsquared:
            data = [
                {
                    condition_no: model_fit_results.condition_number
                    , adjusted_rsquared: model_fit_results.rsquared_adj
                }
            ]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

    # now compute the mean score over all k-folds
    if scoring_method == mse_train_and_mse_test:
        mean_mse_train = scores_df[mse_train].mean()
        mean_mse_test = scores_df[mse_test].mean()
        mean_cv_score = (mean_mse_train, mean_mse_test)

    elif scoring_method == rmse_train_and_rmse_test:
        mean_rmse_train = scores_df[rmse_train].mean()
        mean_rmse_test = scores_df[rmse_test].mean()
        mean_cv_score = (mean_rmse_train, mean_rmse_test)

    elif scoring_method == mse_and_delta_mse:
        mean_mse = scores_df[mse].mean()
        mean_delta_mse = scores_df[delta_mse].mean()
        mean_cv_score = (mean_mse, mean_delta_mse)

    elif scoring_method == rmse_and_delta_rmse:
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_rmse, mean_delta_rmse)

    elif scoring_method == adjusted_rsquared:
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        mean_cv_score = mean_adj_rsq

    elif scoring_method == condition_no:
        mean_cond_no = scores_df[condition_no].mean()
        mean_cv_score = mean_cond_no
    
    elif scoring_method == condition_no_and_adjusted_rsquared:
        mean_cond_no = scores_df[condition_no].mean()
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        mean_cv_score = (mean_cond_no, mean_adj_rsq)

    elif scoring_method == condition_no_and_rmse_and_delta_rmse:
        mean_cond_no = scores_df[condition_no].mean()
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_cond_no, mean_rmse, mean_delta_rmse)

    elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
        mean_cond_no = scores_df[condition_no].mean()
        mean_rsq = scores_df[rsquared].mean()
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        pvals_df = pd.DataFrame(columns=feat_combo)
        for idx, row in  scores_df.iterrows():
            list_pvals = list(row[pvals].values[1:])
            pvals_df = pvals_df.append(pd.Series(list_pvals, index=pvals_df.columns), ignore_index=True, sort=False)
        mean_pvals = []
        for idx, _ in enumerate(feat_combo):
            mean_pvals.append(pvals_df.iloc[:, idx].mean())
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_cond_no, mean_rsq, mean_adj_rsq, mean_pvals, mean_rmse, mean_delta_rmse)

    return (X_train, X_test, y_train, y_test, mean_cv_score)

# this method is now deprecated and is here for reference only
# use cv_selection_dp() instead
def cv_selection_exhaustive(
    X
    , y
    , folds=5
    , reverse=False
    , smargs=None):

    scores_df = pd.DataFrame(columns=['n_features', 'features', condition_no, rsquared, adjusted_rsquared, pvals, rmse, delta_rmse])

    target_cond_no = None
    if smargs is not None:
        target_cond_no = smargs['cond_no']
    if target_cond_no is None:
        target_cond_no = 1000 #default definition of "non-colinearity" used by statsmodels - see https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults.summary

    cv_feat_combo_map, _ = cv_build_feature_combinations(X, reverse=reverse)

    if cv_feat_combo_map is None:
        return
    
    base_feature_set = list(X.columns)
    n = len(base_feature_set)
    
    best_feat_combo = []
    best_score = None
    
    for _, list_of_feat_combos in cv_feat_combo_map.items():
        n_choose_k = len(list_of_feat_combos)
        k = len(list_of_feat_combos[0])
        s_n_choose_k = "{} \\choose {}"
        display(HTML("Cross-validating ${}={}$ combinations of {} features (out of {}) over {} folds...".format("{" + s_n_choose_k.format(n, k) + "}", n_choose_k, k, n, folds)))

        for feat_combo in list_of_feat_combos:           
            feat_combo = list(feat_combo)

            _, _, _, _, score = cv_score(
                X
                , y
                , feat_combo
                , folds
                , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
            )

            # now determine if this score is best
            is_in_conf_interval = False not in [True if pval >= 0.0 and pval <= 0.05 else False for pval in score[3]]
            is_non_colinear = score[0] <= target_cond_no
            if is_non_colinear and is_in_conf_interval and (best_score is None or (score[1] > best_score[1] and score[2] > best_score[2])):
                best_score = score
                best_feat_combo = feat_combo
                print("new best {} score: {}, from feature-set combo: {}".format(condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse, best_score, best_feat_combo))
                data = [
                    {
                        'n_features': len(feat_combo)
                        , 'features': feat_combo
                        , condition_no: score[0]
                        , rsquared: score[1]
                        , adjusted_rsquared: score[2]
                        , pvals: score[3]
                        , rmse: score[4]
                        , delta_rmse: score[5]
                    }
                ]
                scores_df = scores_df.append(data, ignore_index=True, sort=False)
    
    print_df(scores_df)

    display(HTML("<h4>cv_selected best {} = {}</h4>".format(condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse, best_score)))
    display(HTML("<h4>cv_selected best feature-set combo ({} of {} features):{}<h/4>".format(len(best_feat_combo), len(base_feature_set), best_feat_combo)))
    display(HTML("<h4>starting feature-set:{}</h4>".format(base_feature_set)))
    to_drop = list(set(base_feature_set).difference(set(best_feat_combo)))
    display(HTML("<h4>cv_selection suggests dropping {}.</h4>".format(to_drop if len(to_drop)>0 else "<i>no features</i> from {}".format(base_feature_set))))

    return (scores_df, best_feat_combo, best_score, to_drop)

# please see Appendex.ipynb for reference, documentation, and explanation
def cv_selection_dp(
    X
    , y
    , folds=5
    , reverse=False
    , smargs=None):

    cols = ['n_features', 'features', condition_no, rsquared, adjusted_rsquared, pvals, rmse, delta_rmse]
    scores_df = pd.DataFrame(columns=cols)

    target_cond_no = None
    if smargs is not None:
        target_cond_no = smargs['cond_no']
    if target_cond_no is None:
        target_cond_no = 1000 #default definition of "non-colinearity" used by statsmodels - see https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults.summary

    cv_feat_combo_map, _ = cv_build_feature_combinations(X, reverse=reverse)

    if cv_feat_combo_map is None:
        return
    
    base_feature_set = list(X.columns)
    n = len(base_feature_set)
    
    best_feat_combo = []
    best_score = None

    best_feat_combos = []
    
    for _, list_of_feat_combos in cv_feat_combo_map.items():
        n_choose_k = len(list_of_feat_combos)
        k = len(list_of_feat_combos[0])
        depth = k-1
        s_n_choose_k = "{} \\choose {}"
        display(
            HTML(
                "<p><br><br>Cross-validating ${}={}$ combinations of {} features (out of {}) over {} folds using score <b>{}</b> and target cond. no = {}...".format(
                    "{" + s_n_choose_k.format(n, k) + "}"
                    , n_choose_k
                    , k
                    , n
                    , folds
                    , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
                    , target_cond_no
                )
            )
        )

        n_discarded = 0
        n_met_constraints = 0

        for feat_combo in list_of_feat_combos:           
            feat_combo = list(feat_combo)

            closest_prior_depth = min(len(best_feat_combos)-1, depth-1)
            if depth > 0 and closest_prior_depth >= 0:
                last_best_feat_combo = best_feat_combos[closest_prior_depth]
                last_best_feat_combo_in_current_feat_combo = set(last_best_feat_combo).issubset(set(feat_combo))
                if last_best_feat_combo_in_current_feat_combo:
                    #print("depth is {}, best_feat_combos[last_saved_depth]: {}".format(depth, best_feat_combos[last_saved_depth]))
                    #print("feat_combo: {}".format(feat_combo))
                    #print("best_feat_combos[last_saved_depth] in feat_combo: {}".format(last_best_feat_combo_in_current_feat_combo))
                    pass
                else:
                    n_discarded += 1
                    #display(HTML("DISCARDED feature-combo {} since it is not based on last best feature-combo {}; discarded so far: {}".format(feat_combo, last_best_feat_combo, n_discarded)))
                    continue

            _, _, _, _, score = cv_score(
                X
                , y
                , feat_combo
                , folds
                , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
            )

            # now determine if this score is best
            is_in_conf_interval = False not in [True if pval >= 0.0 and pval <= 0.05 else False for pval in score[3]]
            is_non_colinear = score[0] <= target_cond_no
            if is_non_colinear and is_in_conf_interval and (best_score is None or (score[1] > best_score[1] and score[2] > best_score[2])):
                n_met_constraints += 1
                best_score = score
                best_feat_combo = feat_combo
                if len(best_feat_combos) < k:
                    best_feat_combos.append(feat_combo)
                else:
                    best_feat_combos[depth] = feat_combo
                print("new best {} score: {}, from feature-set combo: {}".format(condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse, best_score, best_feat_combo))
                data = [
                    {
                        'n_features': len(feat_combo)
                        , 'features': feat_combo
                        , condition_no: score[0]
                        , rsquared: score[1]
                        , adjusted_rsquared: score[2]
                        , pvals: score[3]
                        , rmse: score[4]
                        , delta_rmse: score[5]
                    }
                ]
                mask = scores_df['n_features']==k
                if len(scores_df.loc[mask]) == 0:
                    scores_df = scores_df.append(data, ignore_index=True, sort=False)
                else:
                    keys = list(data[0].keys())
                    replacement_vals = list(data[0].values())
                    scores_df.loc[mask, keys] = [replacement_vals]
        
        if n_discarded > 0:
            display(HTML("<p>DISCARDED {} {}-feature combinations that were not based on prior optimal feature-combo {}".format(n_discarded, k, last_best_feat_combo)))
        if n_met_constraints > 0:
            display(HTML("<p>cv_selection chose the best of {} {}-feature combinations that met the constraints (out of {} considered)".format(n_met_constraints, k, n_choose_k - n_discarded)))
        if n_choose_k - n_discarded - n_met_constraints > 0:
            display(HTML("<p>{} {}-feature combinations (out of {} considered) failed to meet the constraints<p><br><br>".format(n_choose_k - n_discarded - n_met_constraints, k, n_choose_k - n_discarded)))
    
    display(HTML("<h2>Table of cv_selected Optimized Feature Combinations</h2>"))
    print_df(scores_df)

    display(HTML("<h4>cv_selected best {} = {}</h4>".format(condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse, best_score)))
    display(
        HTML(
            "<h4>cv_selected best feature-set combo ({} of {} features) {} based on {} scoring method with target cond. no. {}<h/4>".format(
                len(best_feat_combo)
                , len(base_feature_set)
                , best_feat_combo
                , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
                , target_cond_no
            )
        )
    )
    display(HTML("<h4>starting feature-set:{}</h4>".format(base_feature_set)))
    to_drop = list(set(base_feature_set).difference(set(best_feat_combo)))
    display(HTML("<h4>cv_selection suggests dropping {}.</h4>".format(to_drop if len(to_drop)>0 else "<i>no features</i> from {}".format(base_feature_set))))

    return (scores_df, best_feat_combo, best_score, to_drop)

def lin_reg_model(
    df
    , target
    , fn_feature_selection=None
    , title="Linear Regression Model"):

    s_html = "<h1>{}</h1><ul>".format(title)
    s_html += "<li><b>target</b>: {}</li>".format(target)
    s_html += "<li><b>starting feature set</b>: {}</li>".format(df.drop(target, axis=1).columns)
    s_html += "</ul>"
    display(HTML(s_html))
    
    data_fin_df = df.copy()
    
    y = data_fin_df[[target]]
    X = data_fin_df.drop([target], axis=1)
    
    if fn_feature_selection is not None:
        print("Feature selection method: {}".format(fn_feature_selection))
        sel_features, score, to_drop = fn_feature_selection(X, y)
    else:
        sel_features = list(X.columns)

    f = target + '~' + "+".join(sel_features)
    print("\nformula: {}".format(f))
    
    X_train, X_test, y_train, y_test, mean_cv_score = cv_score(X, y, sel_features, scoring_method=rmse_train_and_rmse_test)

    data_fin_df = pd.concat([y_train, X_train], axis=1, join='inner').reset_index()
    model = ols(formula=f, data=data_fin_df)

    return (sel_features, X_train, X_test, y_train, y_test, mean_cv_score[0], mean_cv_score[1], model)

def model_fit_summary(
    df
    , sel_features
    , target
    , model
    , train_score
    , test_score
    , mv_r_sq_th
    , mv_delta_score_th
    , mv_bad_vif_ratio_th
    , display_regress_diagnostics=False
    , display_infl_plot=False):

    # get results of OLS fit from previously computed model
    model_fit_results = model.fit()
    #model_fit_results = model.fit(cov_type='HC0')
    #model_fit_results = model.fit(cov_type='HC1')
    #model_fit_results = model.fit(cov_type='HC2')
    #model_fit_results = model.fit(cov_type='HC3')
    
    delta_score = abs(test_score - train_score)
    valid_r_sq = model_fit_results.rsquared >= mv_r_sq_th

    # for each feature, compute VIF and optionally display QQ plot
    if display_regress_diagnostics:
        display(HTML("<h3>Regression Diagnostics</h3>"))
    good_vifs = []
    bad_vifs = []
    for idx, feat in enumerate(sel_features):        
        v = feature_regression_summary(df[sel_features], idx, model_fit_results, display_regress_diagnostics)
        if v <= 10:
            good_vifs.append((feat, round(v, 2), model_fit_results.pvalues[sel_features.index(feat)]))
        else:
            bad_vifs.append((feat, round(v, 2), model_fit_results.pvalues[sel_features.index(feat)]))    
    # order good VIF features by p-value
    good_vifs = sorted(good_vifs, key=lambda good_vif: good_vif[2])    
    # order bad VIF features by VIF
    bad_vifs = sorted(bad_vifs, key=lambda bad_vif: bad_vif[1], reverse=True)
    good_vif_ratio = len(good_vifs)/len(sel_features)
    bad_vif_ratio = 1 - good_vif_ratio
    
    # VIF/colinearity summary
    display(HTML("<h3>VIF Summary</h3>"))
    s_html = "<br><b>'GOOD' FEATURES</b> (<i>with LOW COLINEARITY, VIF <= 10</i>), ordered by favorable (increasing) p-val:<ol>"
    for good_vif in good_vifs:
        s_html += "<li><b>{:30}</b>: p-val $= {}$</li>".format(good_vif[0], good_vif[2])
    s_html += "</ol>"
    display(HTML(s_html))    
    s_html = "<br><b>'BAD' FEATURES</b> (<i>with HIGH COLINEARITY, VIF > 10</i>), ordered by unfavorable (decreasing) VIF:<ol>"
    for bad_vif in bad_vifs:
        s_html += "<li><b>{:30}</b>: VIF $= {}$</li>".format(bad_vif[0], bad_vif[1])
    s_html += "</ol>"
    display(HTML(s_html)) 
    good_vif_ratio = len(good_vifs)/len(sel_features)
    display(HTML("<br><b>{}% of features are 'GOOD'.<br>".format(round(good_vif_ratio*100,2))))
    plot_corr(df[[target] + sel_features])
    
    # always displays QQ plot of residuals and density plots of target 
    fig = plt.figure(figsize=(15, 5) if display_infl_plot else (10, 5))
    axes = fig.subplots(1, 3 if display_infl_plot else 2)
    sm.graphics.qqplot(model_fit_results.resid, dist=stats.norm, line='45', fit=True, ax=axes[0])
    sns.distplot(df[target], ax=axes[1])
    if display_infl_plot:   
        sm.graphics.influence_plot(model_fit_results, ax=axes[2])
    fig.tight_layout()
    plt.show();
    
    # display the OLS summary
    display(HTML(model_fit_results.summary().as_html()))
    
    return (model_fit_results, good_vifs, bad_vifs)

def mfrs_comparison(models, mfrs, scores_dict, titles=["Previous", "Current"]):
    fig = plt.figure(figsize=(10, 5))
    axes = fig.subplots(1, len(mfrs))
    for idx, mfr in enumerate(mfrs):
        sm.graphics.qqplot(mfr.resid, dist=stats.norm, line='45', fit=True, ax=axes[idx])
        axes[idx].set_title(titles[idx], fontsize='large')
    plt.suptitle("Model Comparison", fontsize='x-large')
    #fig.tight_layout()
    plt.show();

    s_html = "<h2>Summary</h2><ol>"

    # features include in models
    s_html += "<li><b>Indep.</b>:<ol>"
    for idx, model in enumerate(models):   
        s_html += "<li>{}: {}</li>".format(titles[idx], model.exog_names[1:])
    s_html += "</ol></li>"

    # "delta" features
    if len(models[1].exog_names) < len(models[0].exog_names):
        feats_dropped = list_difference(models[0].exog_names, models[1].exog_names)
        s_html += "<li><font color='red'>DROPPED</font> Indep.: {}</li>".format(feats_dropped)
    elif len(models[1].exog_names) > len(models[0].exog_names):
        feats_added = list_difference(models[1].exog_names, models[0].exog_names)
        s_html += "<li><font color='green'>ADDED</font> Indep.: {}</li>".format(feats_added)
    else:
        s_html += "<li>NO DIFFERENCE in Indep. (perhaps transformations/scalings of indeps. differ?)</li>"

    # r squared
    s_html += "<li>$R^2$:<ol>"
    for idx, mfr in enumerate(mfrs):   
        s_html += "<li>{}: {}</li>".format(titles[idx], mfr.rsquared)
    s_html += "</ol></li>"

    # delta r squared
    d_rsq = mfrs[1].rsquared - mfrs[0].rsquared
    s_html += "<li>$\Delta R^2$: {} ({})</li>".format(
        d_rsq,
         "<font color='green'>{}% better/increase</font>".format(round((mfrs[1].rsquared/mfrs[0].rsquared)*100, 2)) if d_rsq > 0 \
                else ("<font color='red'>{}% worse/reduction</font>".format(round((1-mfrs[1].rsquared/mfrs[0].rsquared)*100, 2)) if d_rsq < 0 else "no change")
    )

    # adj r squared
    s_html += "<li><b>Adjusted $R^2$</b>:<ol>"
    for idx, mfr in enumerate(mfrs):   
        s_html += "<li>{}: {}</li>".format(titles[idx], mfr.rsquared_adj)
    s_html += "</ol></li>"

    # delta r sq - adj r sq
    d_arsq_0 = mfrs[0].rsquared - mfrs[0].rsquared_adj
    d_arsq_1 = mfrs[1].rsquared - mfrs[1].rsquared_adj
    d_d_arsq = d_arsq_1 - d_arsq_0
    s_html += "<li><b>$\Delta$ ($R^2 -$ Adj. $R^2$)</b>: {} ({})</li>".format(
        d_d_arsq
        , "<font color='green'>{}% better/reduction</font>".format(round((d_arsq_0/d_arsq_1)*100, 2)) if d_d_arsq < 0 \
            else ("<font color='red'>{}% worse/increase</font>".format(round((d_arsq_1/d_arsq_0)*100, 2)) if d_d_arsq > 0 else "no change")
        )

    # scoring method
    s_html += "<li><b>{}</b>: <ol>".format(scores_dict['method'])
    scores = scores_dict['scores']
    d_score_0 = abs(scores[0][0] - scores[0][1])
    s_html += "<li>{}: {} $\\implies \Delta: {}$</li>".format(titles[0], scores[0], d_score_0)
    d_score_1 = abs(scores[1][0] - scores[1][1])
    s_html += "<li>{}: {} $\\implies \Delta: {}$</li>".format(titles[1], scores[1], d_score_1)
    s_html += "</ol></li>"

    # delta scoring method
    d_d_score = d_score_1 - d_score_0
    s_html += "<li><b>$\Delta \Delta$ {}</b>: {} ({})<ol>".format(
        scores_dict['method']
        , d_d_score
        , "<font color='green'>{}% better/reduction</font>".format(round((d_score_0/d_score_1)*100, 2)) if d_d_score < 0 \
            else ("<font color='red'>{}% worse/increase</font>".format(round((d_score_1/d_score_0)*100, 2)) if d_d_score > 0 else "no change")
    )    
    s_html += "</ol></li>"

    # condition no.
    s_html += "<li><b>Condition No.</b>:<ol>"
    for idx, mfr in enumerate(mfrs):   
        s_html += "<li>{}: {}</li>".format(titles[idx], mfr.condition_number)
    s_html += "</ol></li>"

    # delta cond. no.
    d_condno = mfrs[1].condition_number - mfrs[0].condition_number
    s_html += "<li><b>$\Delta$ Condition No</b>: {} ({})</li>".format(
        d_condno
        , "<font color='green'>{}% better/reduction</font>".format(round((mfrs[0].condition_number/mfrs[1].condition_number)*100, 2)) if d_condno < 0 \
            else ("<font color='red'>{}% worse/increase</font>".format(round((mfrs[1].condition_number/mfrs[0].condition_number)*100, 2)) if d_condno > 0 else "no change")
    )

    s_html += "</ol>"
    display(HTML(s_html))

def scatter_plots(df, target):
    df_minus_target = df.drop(target, axis=1)
    
    s_html = "<h3>Scatter Plots:</h3><ul>"
    s_html += "<li><b>target</b>: {}</li>".format(target)
    s_html += "<li><b>feature set</b>: {}</li>".format(df_minus_target.columns)
    s_html += "</ul>"
    display(HTML(s_html))
    
    r_w = 20
    r_h = 4

    c_n = 4 if len(df_minus_target.columns) >= 4 else len(df_minus_target.columns)
    r_n = len(df_minus_target.columns)/c_n
    r_n = int(r_n) + (1 if r_n % int(r_n) != 0 else 0)

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)
    unused = list(range(0, len(axes.flatten()))) if len(df_minus_target.columns) > 1 else [0]
    
    for index, feat in enumerate(df_minus_target):
        ax = fig.add_subplot(r_n, c_n, index+1)
        #plt.scatter(df[feat], df[target], alpha=0.2)
        sns.scatterplot(df[feat], df[target], alpha=0.2, ax=ax)
        plt.xlabel(feat)
        plt.ylabel(target)
        unused.remove(index)

    flattened_axes = axes.flatten() if len(df_minus_target.columns) > 1 else [axes]
    for u in unused:
        fig.delaxes(flattened_axes[u])

    fig.tight_layout()
    plt.show();

def scatterplot_comparison(X1, X2, y1, y2, title1, title2):    
    s_html = "<h3>Scatterplot Comparisons:</h3>"
    display(HTML(s_html))
    
    n_feats = len(X1.columns)
    
    r_w = 2*plot_edge
    r_h = plot_edge
    
    c_n = 2
    r_n = n_feats       

    fig = plt.figure(figsize=(r_w, r_h*r_n))

    axes = fig.subplots(r_n, c_n)

    for index in range(0, n_feats):
        ax1 = fig.add_subplot(r_n, c_n, 2*index+1)
        sns.scatterplot(X1.iloc[:, index], y1.iloc[:, 0], alpha=0.2, ax=ax1)
        ax1.set_ylabel(y1.columns[0])
        ax1.set_xlabel(X1.columns[index])
        ax1.set_title(title1)
        ax2 = fig.add_subplot(r_n, c_n, 2*index+2)
        sns.scatterplot(X2.iloc[:, index], y2.iloc[:, 0], alpha=0.2, ax=ax2)
        ax2.set_ylabel(y2.columns[0])
        ax2.set_xlabel(X2.columns[index])
        ax2.set_title(title2)

    fig.tight_layout()
    plt.show();

def plot_feature_importance(mfr, title):
    feature_coeffs = np.absolute(mfr.params[1:]).sort_values(ascending=False)
    feat_names = list(feature_coeffs.index)
    feat_vals = list(feature_coeffs.values)
    fig = plt.figure(figsize=(12, 8))
    plt.title("Relative Importance of Features: {}".format(title))
    sns.barplot(x=feat_names, y=feat_vals, color='red')
    fig.tight_layout()
    plt.show();