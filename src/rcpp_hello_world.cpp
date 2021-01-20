#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>


// [[Rcpp::export]]
double logdmvt_Rcpp(const arma::vec& x, const arma::mat& S) {
    double twoa0 = 0.02; // prior is a_0 = 0.01, b_0 = 0.01 so df = 0.02
    arma::uword n = x.n_elem;
    arma::mat St = chol(S);
    arma::vec B = diagvec(St);
    double log_det_S = sum(log(B));
    double result;
    double P = -0.5 * log_det_S;
    arma::vec y = solve(arma::trimatl(St.t()), x);
    result = arma::as_scalar(P - ((twoa0 + n) * 0.5) * log(1.0 + (1.0 / twoa0) *
             std::pow(norm(y),2)));
    return result;
}


// [[Rcpp::export]]
arma::mat generate_rows(const double df, 
    const arma::mat& design_mat, const arma::uword n_bss,
    const arma::uword n_parents) {
    
    arma::vec log_bf(n_bss); // Vektor att spara resultaten från resp bss i
    const int n_obs = design_mat.n_rows;

    // Generate dependent variable
    Rcpp::NumericVector randomT = Rcpp::rt(n_obs*n_bss, df);
    double scale = std::sqrt( (df-2.0)/df );
    arma::vec randT = randomT*scale;
    arma::mat ySim(randT.begin(), n_obs, n_parents, true);
    ySim.each_col() += design_mat.col(0); // y = mu + epsilon

    // Prepare vectors
    arma::vec x2 = design_mat.col(1); //kovariater för modell 1
    arma::vec x3 = design_mat.col(2); //kovariater för modell 2
    arma::vec y_i(n_obs);

    // Used to generate bootstrap samples
    arma::uvec idx = arma::linspace<arma::uvec>(0, n_obs-1, n_obs); 
    arma::vec prob(n_obs, arma::fill::zeros);
    prob += 1.0/n_obs;

    arma::mat resvec(n_parents, 6, arma::fill::zeros);
    double all = n_bss;

    // Intialization
    arma::vec y_b(n_obs), x2_b(n_obs), x3_b(n_obs);
    arma::uvec bindx;
    double log_ml1, log_ml2, lbf_true;

    
    
    for (arma::uword i = 0; i < n_parents; ++i) {

        y_i = ySim.col(i);

        // Calculate the true lbf for this parent

        arma::mat UnityMatrix;
        arma::mat S1 = UnityMatrix.eye(n_obs, n_obs) + x2 * x2.t();
        arma::mat S2 = UnityMatrix.eye(n_obs, n_obs) + x3 * x3.t();
        log_ml1 = logdmvt_Rcpp(y_i, S1);
        log_ml2 = logdmvt_Rcpp(y_i, S2);
        lbf_true = log_ml1 - log_ml2;

        for (arma::uword j = 0; j < n_bss; ++j) {
            bindx = Rcpp::RcppArmadillo::sample(idx, n_obs, true, prob);
            y_b = y_i(bindx);
            x2_b = x2(bindx);
            x3_b = x3(bindx);
            arma::mat UnityMatrix;
            arma::mat S1 = UnityMatrix.eye(n_obs, n_obs) + x2_b * x2_b.t();
            arma::mat S2 = UnityMatrix.eye(n_obs, n_obs) + x3_b * x3_b.t();
            log_ml1 = logdmvt_Rcpp(y_b, S1);
            log_ml2 = logdmvt_Rcpp(y_b, S2);

            log_bf(j) = log_ml1 - log_ml2;
        }

    resvec(i, 3) = lbf_true;
    resvec(i, 4) = arma::var(log_bf);
    resvec(i, 5) = (sum(abs(log_bf)>5.0)/all);

    }

    resvec.col(0) += n_bss;
    resvec.col(1) += n_obs;
    resvec.col(2) += df;

    return resvec;
}