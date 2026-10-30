#ifndef KENT_ESTIMATOR_H
#define KENT_ESTIMATOR_H
// MIS Laboratory, University of Picardy Jules Verne
// Author: Maxime Robic maxime.robic@u-picardie.fr
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <numeric>
#include <iostream>
#include <cmath>

#include "angle.hpp"


using namespace cv;

/// @brief Recursive Bayesian filter on the unit sphere based on the Kent (FB5) distribution.
///
/// The state is Kent(mu = est_u_, kappa = est_k_, beta = est_beta_, Gamma = [xi1 xi2 mu]),
/// with density proportional to exp(kappa mu^T x + x^T A x), A = Gamma Lambda Gamma^T,
/// Lambda = diag(beta, -beta, 0). Measurements are bearings Zk on the sphere together with
/// a flow point Fk, whose tangent direction at Zk gives the measured major axis xi1.
class KentEstimator
{
private:
  Angle ang_;

  double R(double k, double beta)
    {
    if(k-2*beta <= 1e-6){ 
      std::cout <<" Warning: Kent R function denominator too small (k<=2beta), computing R from VMF" << std::endl;
      return R_VMF(k);}
    else{
    return 1.0 - k/((k-2*beta)*(k+2*beta));
    }
  }
  double Q(double k, double beta)
    {
    if(k-2*beta <= 1e-6){ 
      std::cout <<" Warning: Kent Q function denominator too small (k<=2beta), setting Q to 0" << std::endl;
      return 0;}
    else{
    return  4*beta/((k-2*beta)*(k+2*beta));
    }
    }
  void convolveKent(double R_1, double Q_1, double R_2, double Q_2, double &k_conv, double &beta_conv)
    {
    double R_conv = R_1 * R_2;
    double Q_conv = Q_1 + Q_2;
    double denom = 4*pow(1-R_conv,2) - pow(Q_conv,2);
    if (denom < 1e-6) { k_conv = 1, beta_conv = 0;
      std::cout <<" Warning: Kent convolution denominator too small, setting beta_conv to 0 kappa_conv to 1" << std::endl;

    }
    else{
    k_conv = 4*(1-R_conv)/(4*pow(1-R_conv,2) - pow(Q_conv,2));
    beta_conv = Q_conv/(4*pow(1-R_conv,2) - pow(Q_conv,2));}
    }
      double R_VMF(double k)
    {
    return 1 / tanh(k) - 1 / k;
    }


  cv::Mat computeA(double beta_s, Point3d Z_s, double phi_s){
        cv::Mat A_s = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat Lambda_s = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat Gamma_s = cv::Mat::zeros(3, 3, CV_64F);

        Lambda_s = (cv::Mat_<double>(3,3) <<
        beta_s, 0,   0,
        0,  -beta_s, 0,
        0,    0,   0
        );

        double theta = acos(min(max(Z_s.z,-1.0),1.0));
        cv::Mat R_mu = cv::Mat::eye(3, 3, CV_64F);
        double epsilon = 1e-6;

        if (std::abs(theta) > epsilon) {
            cv::Vec3d Zp(0.0, 0.0, 1.0);
            cv::Vec3d u = Zp.cross(Z_s);
            u = u / cv::norm(u);
            cv::Vec3d axis_angle = u * theta;
            cv::Rodrigues(axis_angle, R_mu);
        }

        cv::Mat R_phi = (cv::Mat_<double>(3,3) <<
            cos(phi_s), -sin(phi_s), 0,
            sin(phi_s),  cos(phi_s), 0,
            0,         0,        1
);

      cv::Mat Gamma_meas = R_mu.t() * R_phi;

      cv::Point3d xi1_s(Gamma_meas.at<double>(0,0), Gamma_meas.at<double>(1,0), Gamma_meas.at<double>(2,0));
      cv::Point3d xi2_s(Gamma_meas.at<double>(0,1), Gamma_meas.at<double>(1,1), Gamma_meas.at<double>(2,1));

        Gamma_s = (cv::Mat_<double>(3,3) <<
        xi1_s.x,    xi2_s.x,   Z_s.x,
        xi1_s.y,    xi2_s.y,   Z_s.y,
        xi1_s.z,    xi2_s.z ,  Z_s.z
        );
  
        A_s = Gamma_s * Lambda_s * Gamma_s.t();

        return A_s;
  }

  cv::Mat computeAFromFlow(double beta_s, Point3d Z_s, Point3d F_s){

        cv::Mat Lambda_s = (cv::Mat_<double>(3,3) <<
        beta_s, 0,   0,
        0,  -beta_s, 0,
        0,    0,   0
        );

        cv::Mat Gamma_s = computeGammaFromFlow(Z_s,F_s);

        cv::Mat A_s = Gamma_s * Lambda_s * Gamma_s.t();

        return A_s;
  }



public:
  Point3d est_u_;
  Point3d est_xi_1;
  Point3d est_xi_2;
  double est_k_, est_beta_;

  // --- A_sum (exact fused matrix) vs A_opt (Kent-valid projection) comparison, refreshed each update() ---
  double aErrFrob_    = 0.0;  // ||A_sum - A_opt||_F               (absolute Frobenius error)
  double aSumFrob_    = 0.0;  // ||A_sum||_F                       (magnitude, for the relative error)
  double aErrFrobRel_ = 0.0;  // ||A_sum - A_opt||_F / ||A_sum||_F (scale-invariant relative error)
  double aTangResid_  = 0.0;  // ||A_sum u - (u^T A_sum u) u||     (how far est_u_ is from an eigenvector of A_sum)
  double aBetaSum_    = 0.0;  // beta implied by A_sum = lambda_max(A_sum)  (unprojected ovalness)
  double aBetaOpt_    = 0.0;  // beta implied by A_opt = lambda_max(A_opt)  (== est_beta_)
  bool   aValid_      = false;// true if the metrics above were refreshed this update (A_opt finite)
  long   aUpdateCount_ = 0;   // increments each time the metrics are refreshed (for one-row-per-update logging)

  cv::Mat A = cv::Mat::zeros(3, 3, CV_64F);
  cv::Mat Lambda = cv::Mat::zeros(3, 3, CV_64F);
  cv::Mat Gamma = cv::Mat::zeros(3, 3, CV_64F);
  cv::Mat Gamma_1 = cv::Mat::zeros(3, 3, CV_64F);

  double k_m_, beta_m_;
  double k_s_, beta_s_;

  Point3d lastFxi1;
  bool initialized_=false;
  std::vector<cv::Mat> rot_incr_;
  std::vector<double> dt_vec_;
  int nInit = 0;
  int noZcount = 0;

  Point3d measureBearing;
  Point3d measureHeading;

  // ---- Kent-aware data association (robustness to dynamic distractors) ----
  // false == bearing-only nearest-neighbor association. KentTracker enables it by default.
  bool   use_heading_assoc_    = false; // gate/score associations with the tracked heading
  double assoc_bearing_gate_   = 15.0;  // deg, hard gate on bearing geodesic distance (as before)
  double assoc_heading_gate_   = 60.0;  // deg, hard gate on flow-heading discrepancy
  double assoc_heading_weight_ = 1.0;   // weight of the heading term in the association cost (fixed-weight mode; best config)
  double assoc_beta_activate_  = 5.0;   // trust heading only once the Kent is anisotropic enough
  // Adaptive association cost: weight each term by the distribution's own confidence.
  //   cost = (kappa/kappa_max)*db + (beta/beta_max)*dh
  // => sharper bearing (high kappa) makes the bearing gate dominate; stronger anisotropy (high beta)
  //    gives the heading gate more say. Enabled by the --adaptive flag of KentTracker.
  bool   adaptive_assoc_       = false;
  double beta_max_             = 1000.0; // hard cap on beta (numerical); NOT the adaptive normalizer
  // Adaptive-weight normalizers = RUNNING MAX of kappa/beta actually observed in the sequence, so the
  // weights span [0,1] over the real operating range (not squashed by the numerical caps).
  double kappaSeenMax_        = 1e-6;
  double betaSeenMax_         = 1e-6;
  // Association diagnostics (logged by KentTracker)
  int    assocDivergeCount_  = 0;      // #updates Kent-assoc chose a different meas than bearing-NN
  int    headingRejectCount_ = 0;       // #candidate associations rejected by the heading gate
  bool   lastAssocDiverged_  = false;
  double lastHeadingDisc_    = 0.0;     // rad, heading discrepancy of the chosen measurement
  // Cost breakdown of the selected measurement
  double lastAssocCost_      = 0.0;     // total association cost of the chosen measurement
  double lastAssocDb_        = 0.0;     // rad, bearing term db of the chosen measurement
  double lastAssocDh_        = 0.0;     // rad, heading term dh of the chosen measurement
  double lastBearingW_       = 1.0;     // bearing weight used this step (adaptive: kappa/kappa_max)
  double lastHeadingW_       = 1.0;     // heading weight used this step (adaptive: beta/beta_max)
  // ---- Reinitialization control (coast through distractor gaps) ----
  // These are INPUT thresholds that trigger a reinit; nInit above is the OUTPUT count of reinits.
  double reinit_kappa_min_   = 50.0;    // reinit when est_k_ falls below this (usually the first to fire)
  int    reinit_miss_limit_  = 100;     // reinit after this many consecutive missed associations
  double kappa_max_          = 2000.0;  // cap on est_k_ to prevent numerical runaway (healthy ~500)

  KentEstimator()
    {
    initialized_ = false;
    }

  ~KentEstimator()
    {
    }

  void initialize(Point3d u, Point3d xi1, Point3d xi2, double k, double beta)
    {

    double norm_u = cv::norm(u);
    if (norm_u < 1e-8) {
        std::cout << "Initialization failed: est_u_ norm too small." << std::endl;
        initialized_ = false;
        return;
    }
    u = u * (1.0 / norm_u);


    double norm_xi1 = cv::norm(xi1);
    double norm_xi2 = cv::norm(xi2);
    if (norm_xi1 < 1e-8 || norm_xi2 < 1e-8) {
        std::cout << "Initialization failed: xi1 or xi2 norm too small." << std::endl;
        initialized_ = false;
        return;
    }
    xi1 = xi1 * (1.0 / norm_xi1);
    xi2 = xi2 * (1.0 / norm_xi2);


    if (k <= 0 || beta < 0 || beta > k/2.0) {
        std::cout << "Initialization failed: invalid k or beta." << std::endl;
        initialized_ = false;
        return;
    }
    // Reject a non-finite seed bearing (degenerate flow -> NaN Zk[0]); otherwise est_u_ would stay NaN
    // until kappa decays enough to trigger a reinitialization.
    if (!std::isfinite(u.x) || !std::isfinite(u.y) || !std::isfinite(u.z)) {
        std::cout << "Initialization failed: non-finite seed bearing." << std::endl;
        initialized_ = false;
        return;
    }

    initialized_ = true;
    est_u_ = u;
    est_xi_1 = xi1;
    est_xi_2 = xi2;
    est_k_ = k;
    est_beta_ = beta;
    // reinit the adaptive-weight running maxes so a fresh track re-learns its own kappa/beta scale
    kappaSeenMax_ = k;
    betaSeenMax_  = std::max(beta, 1e-6);
    rot_incr_.clear();
    dt_vec_.clear();    
    
    Lambda = (cv::Mat_<double>(3,3) <<
    beta, 0,   0,
    0,  -beta, 0,
    0,    0,   0
    );
    Gamma = (cv::Mat_<double>(3,3) <<
    est_xi_1.x,    est_xi_2.x,    est_u_.x,
    est_xi_1.y,    est_xi_2.y,    est_u_.y,
    est_xi_1.z,    est_xi_2.z,    est_u_.z
    );

    lastFxi1 = est_xi_1;

    measureBearing = est_u_;
    measureHeading = lastFxi1;

    A = Gamma * Lambda * Gamma.t();

    Gamma_1 = Gamma;

    noZcount=0;
    nInit++;
    std::cout << "Kent filter initialized with position " << est_u_ << " on the sphere" << std::endl;
    }

  /// @brief Prediction step: rotates Gamma by the mean angular velocity estimated from the last
  /// rotation increments, then convolves the state with the Kent process noise (kappa_m, beta_m).
  void predict(double kappa_m, double beta_m, std::vector<Point3d> Zk, double dt)
    {
    // calculate the average rotational speed of the object
    cv::Vec3d mean_rot_vel(0.0, 0.0, 0.0);
    double dt_sum = 0.0;

    if (!rot_incr_.empty())
    {
        cv::Mat mean_R = cv::Mat::eye(3, 3, CV_64F);

         for(int i = 0; i < rot_incr_.size(); i++){
          mean_R = rot_incr_[i]*mean_R;
          dt_sum += dt_vec_[i];
         }

        cv::Rodrigues(mean_R, mean_rot_vel);
        mean_rot_vel = mean_rot_vel/dt_sum;
        cv::Rodrigues(mean_rot_vel*dt, mean_R);

        Gamma = mean_R * Gamma;
        est_xi_1.x = Gamma.at<double>(0,0);
        est_xi_1.y = Gamma.at<double>(1,0);
        est_xi_1.z = Gamma.at<double>(2,0);
        est_xi_2.x = Gamma.at<double>(0,1);
        est_xi_2.y = Gamma.at<double>(1,1);
        est_xi_2.z = Gamma.at<double>(2,1);
        est_u_.x = Gamma.at<double>(0,2);
        est_u_.y = Gamma.at<double>(1,2);
        est_u_.z = Gamma.at<double>(2,2);
        // guard: a NaN Gamma (from an earlier degenerate step) would propagate a non-finite bearing
        if (!std::isfinite(est_u_.x) || !std::isfinite(est_u_.y) || !std::isfinite(est_u_.z)) {
          initialized_ = false;
          std::cout << " Kent predict produced non-finite est_u_, reinitialising" << std::endl;
          return;
        }
    }

    // keep the size of the velocity vector bounded
    if (rot_incr_.size() > 5){
        rot_incr_.erase(rot_incr_.begin());
        dt_vec_.erase(dt_vec_.begin());
        }

    if(Zk.empty() && !rot_incr_.empty())
    {
       // when no measurements remove the oldest velocity
        rot_incr_.erase(rot_incr_.begin());
        dt_vec_.erase(dt_vec_.begin());
    }
   
    double R_est = R(est_k_, est_beta_);
    double Q_est = Q(est_k_, est_beta_);
    double R_m = R(kappa_m, beta_m);
    double Q_m = Q(kappa_m, beta_m);
    convolveKent(R_m, Q_m, R_est, Q_est, est_k_, est_beta_);

    // Guard: convolution can produce non-finite/degenerate concentration -> reinit instead of propagating NaN
    if (!std::isfinite(est_k_) || est_k_ <= 0.0) {
      initialized_ = false;
      std::cout << " Kent predict produced invalid kappa, reinitialising" << std::endl;
      return;
    }
    if (!std::isfinite(est_beta_)) est_beta_ = 0.0;
    est_k_ = std::min(est_k_, kappa_max_);   // cap runaway before it overflows the A-matrix

    Lambda = (cv::Mat_<double>(3,3) <<
        est_beta_, 0,   0,
        0,  -est_beta_, 0,
        0,    0,   0
        );

    A = Gamma * Lambda * Gamma.t();

    if(est_k_<reinit_kappa_min_){ // reinitialize the tracker when the concentration becomes too low
      initialized_=false;
      std::cout<< " Need to reinitialise " << std::endl;
    }
    if(est_beta_>est_k_/2.0){
      initialized_=false;
      std::cout<< " Warning state Kent distribution is not unimodal anymore, need to reinitialise " << std::endl;
    }
  }

  /// @brief Update step: associates one measurement (Zk, Fk) with the predicted state, fuses the
  /// bearing (von Mises-Fisher product) and the ovalness matrices (A_sum), then projects A_sum onto
  /// a Kent-valid matrix A_opt orthogonal to the fused mode.
  /// @param kappa_s, beta_s measurement concentration and ovalness
  /// @param Zk_vec candidate bearings on the unit sphere
  /// @param Fk_vec flow points on the unit sphere associated with each bearing
  /// @param dt time since the last update (s)
void update(double kappa_s, double beta_s, std::vector<Point3d> Zk_vec, std::vector<Point3d> Fk_vec , double dt)
    {
    Point3d Zk, Fk, u_prev = est_u_;
    int min_idx = 0;
    lastAssocDiverged_ = false;
    lastHeadingDisc_   = 0.0;
    if (!Zk_vec.empty())
    {
        // ---- Kent-aware data association ----
        // Candidates must pass a hard bearing gate (as before). When the track is
        // anisotropic enough, they must also agree in flow-heading with the tracked xi1,
        // and the winner minimizes a joint bearing+heading cost. Distractors that share
        // the target's bearing but not its motion direction (blinking lights: no coherent
        // flow; rain: vertical flow; paper bullets: transient off-axis flow) are rejected.
        const double bearing_gate = ang_.degToRad(assoc_bearing_gate_);
        const double heading_gate = ang_.degToRad(assoc_heading_gate_);
        const bool heading_active = use_heading_assoc_ && (est_beta_ > assoc_beta_activate_);
        cv::Vec3d est_xi1_vec(est_xi_1.x, est_xi_1.y, est_xi_1.z);

        // Association-cost weights. Fixed mode: (1.0, assoc_heading_weight_). Adaptive mode: normalize
        // kappa/beta by their RUNNING MAX (updated online each step, reset on reinit) so the weights use
        // the full [0,1] range of the range actually observed for this track.
        if (est_k_    > kappaSeenMax_) kappaSeenMax_ = est_k_;
        if (est_beta_ > betaSeenMax_)  betaSeenMax_  = est_beta_;
        const double aw_b = adaptive_assoc_ ? (est_k_    / std::max(kappaSeenMax_, 1e-6)) : 1.0;
        const double aw_h = adaptive_assoc_ ? (est_beta_ / std::max(betaSeenMax_,  1e-6)) : assoc_heading_weight_;

        int    nn_idx = -1;   double nn_db    = 1e9;  // pure bearing nearest-neighbor (baseline)
        int    best_idx = -1; double best_cost = 1e9; // Kent-aware winner
        double best_dh = 0.0; double best_db = 0.0;

        for (int i = 0; i < (int) Zk_vec.size(); i++)
        {
            double db = acos(min(max(est_u_.dot(Zk_vec.at(i)), -1.0), 1.0)); // bearing geodesic
            if (db > bearing_gate) continue;                                 // hard bearing gate

            // Track the bearing-only nearest neighbor for comparison/logging
            if (db < nn_db) { nn_db = db; nn_idx = i; }

            // Flow-heading discrepancy between the measurement and the tracked heading xi1
            double dh = 0.0;
            if (heading_active) {
                cv::Vec3d Zd(Zk_vec.at(i).x, Zk_vec.at(i).y, Zk_vec.at(i).z);
                cv::Vec3d Fd(Fk_vec.at(i).x, Fk_vec.at(i).y, Fk_vec.at(i).z);
                cv::Vec3d tau = Fd - (Zd.dot(Fd)) * Zd;
                double ntau = cv::norm(tau);
                if (ntau > 1e-6) {                       // only discriminate when a flow direction exists
                    tau = tau / ntau;
                    dh = acos(min(max(tau.dot(est_xi1_vec), -1.0), 1.0));
                    if (dh > heading_gate) { headingRejectCount_++; continue; } // hard heading gate
                }
            }

            double cost = aw_b * db + aw_h * dh;                  // fixed: db + w*dh ; adaptive: running-max weighted
            if (cost < best_cost) { best_cost = cost; best_idx = i; best_dh = dh; best_db = db; }
        }

        if (best_idx < 0) { // no measurement passed the association gates
            std::cout << "No measurements close to the predicted state" << std::endl;
            noZcount++;
            if (noZcount > reinit_miss_limit_) {
                initialized_ = false;
                std::cout << " Need to reinitialise " << std::endl;
                noZcount = 0;
            }
            return;
        }

        min_idx = best_idx;
        lastHeadingDisc_ = best_dh;
        // record the selected measurement's cost breakdown + weights
        lastAssocDb_   = best_db;
        lastAssocDh_   = best_dh;
        lastBearingW_  = aw_b;
        lastHeadingW_  = aw_h;
        lastAssocCost_ = best_cost;
        if (nn_idx >= 0 && nn_idx != best_idx) { lastAssocDiverged_ = true; assocDivergeCount_++; }

        Zk = Zk_vec[min_idx];
        measureBearing = Zk; // associated measurement (heading below is its flow projected on the tangent plane)
        Fk = Fk_vec[min_idx];
        cv::Vec3d tau_k=  Fk - (Zk).dot(Fk) * Zk;
        tau_k = tau_k /cv::norm(tau_k);
   
        cv::Mat Gamma_s = computeGammaFromFlow(Zk,Fk);
        lastFxi1.x = Gamma_s.at<double>(0,0);
        lastFxi1.y = Gamma_s.at<double>(1,0);
        lastFxi1.z = Gamma_s.at<double>(2,0);

        measureHeading = lastFxi1;

        double k_prod = sqrt(pow(est_k_,2) + pow(kappa_s,2) + 2.0 * est_k_ * kappa_s * est_u_.dot(Zk));
        // Guard: degenerate fusion (near-antipodal / non-finite) would produce NaN in est_u_
        if (!std::isfinite(k_prod) || k_prod < 1e-9) {
          initialized_ = false;
          std::cout << " Kent update degenerate (k_prod~0), reinitialising" << std::endl;
          return;
        }
        est_u_ = est_k_ * est_u_ + kappa_s * Zk;
        est_u_ = est_u_ *(1.0 / k_prod);
        est_k_ = std::min(k_prod, kappa_max_);   // cap runaway before it overflows the A-matrix
        // guard: never let a NaN bearing persist (would render as a garbage "opposite" vector until reinit)
        if (!std::isfinite(est_u_.x) || !std::isfinite(est_u_.y) || !std::isfinite(est_u_.z)) {
          initialized_ = false;
          std::cout << " Kent update produced non-finite est_u_, reinitialising" << std::endl;
          return;
        }

        cv::Mat A_sum = A + computeAFromFlow(beta_s, Zk, Fk);

        cv::Vec3d est_u_vec(est_u_.x, est_u_.y, est_u_.z);

        // Kent-valid projection of A_sum: symmetric, traceless and with est_u_ as eigenvector (eigenvalue 0)
        cv::Mat euclMat = 0.5*(est_u_vec.t()*A_sum*est_u_vec);
        double euclScalar = euclMat.at<double>(0,0);
        cv::Mat A_opt = A_sum*(cv::Mat::eye(3, 3, CV_64F)-est_u_vec*est_u_vec.t()) + euclScalar*(cv::Mat::eye(3, 3, CV_64F) + est_u_vec*est_u_vec.t()) - est_u_vec*est_u_vec.t()*A_sum;
    
        // Guard: cv::eigen aborts (SIGABRT) on non-finite input. Bail to reinit instead of crashing.
        if (!cv::checkRange(A_opt)) {
          initialized_ = false;
          std::cout << " Kent A_opt not finite, reinitialising" << std::endl;
          return;
        }
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(A_opt, eigenvalues, eigenvectors);

        // ---- A_sum vs A_opt approximation error ------------------------------------------
        // 1) absolute Frobenius, 2) relative Frobenius, 4) tangency residual, 5) beta = lambda_max
        aErrFrob_ = cv::norm(A_sum - A_opt, cv::NORM_L2);          // ||.||_F for a matrix
        aSumFrob_ = cv::norm(A_sum, cv::NORM_L2);
        aErrFrobRel_ = (aSumFrob_ > 1e-12) ? aErrFrob_ / aSumFrob_ : 0.0;
        {
            // tangency residual: component of A_sum*u orthogonal to u (0 iff u is an eigenvector of A_sum)
            cv::Mat u = (cv::Mat_<double>(3,1) << est_u_.x, est_u_.y, est_u_.z);
            cv::Mat Au = A_sum * u;
            double uAu = u.dot(Au);
            aTangResid_ = cv::norm(Au - uAu * u, cv::NORM_L2);
        }
        aBetaOpt_ = eigenvalues.at<double>(0, 0);                  // lambda_max(A_opt) == est_beta_ (set below)
        if (cv::checkRange(A_sum)) {                               // lambda_max(A_sum) (guard: eigen aborts on non-finite)
            cv::Mat evalSum;
            cv::eigen(A_sum, evalSum);                             // descending -> [0] is largest
            aBetaSum_ = evalSum.at<double>(0, 0);
        } else {
            aBetaSum_ = aBetaOpt_;
        }
        aValid_ = true;
        aUpdateCount_++;
        // ---------------------------------------------------------------------------------

        cv::Mat Gamma_prod = eigenvectors.t();

        cv::Mat Lambda_prod = cv::Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
            Lambda_prod.at<double>(i, i) = eigenvalues.at<double>(0, i);
    
        cv::Point3d mu_temp;
        mu_temp.x = Gamma_prod.at<double>(0,1);
        mu_temp.y = Gamma_prod.at<double>(1,1);
        mu_temp.z = Gamma_prod.at<double>(2,1);
        double sign_mu = est_u_.dot(mu_temp);

        if (sign_mu < 0){//handling numerical instability when mu flips
         Gamma_prod.col(0) = -Gamma_prod.col(0);
         Gamma_prod.col(2) = -Gamma_prod.col(2);
        }

      
        // reorder eigenvectors to Gamma = [xi1 xi2 mu] (swap columns 1 and 2)
        cv::Mat temp_col;
        Gamma_prod.col(2).copyTo(temp_col);
        Gamma_prod.col(1).copyTo(Gamma_prod.col(2));
        temp_col.copyTo(Gamma_prod.col(1));

        Lambda_prod.at<double>(1,1) = Lambda_prod.at<double>(2,2);
        Lambda_prod.at<double>(2,2) = 0.0;

        Gamma_prod.at<double>(0,2) = est_u_.x;
        Gamma_prod.at<double>(1,2) = est_u_.y;
        Gamma_prod.at<double>(2,2) = est_u_.z;

        cv::Vec3d temp_xi1,temp_xi2;
        temp_xi1(0) = Gamma_prod.at<double>(0,0);
        temp_xi1(1) = Gamma_prod.at<double>(1,0);
        temp_xi1(2) = Gamma_prod.at<double>(2,0);
        temp_xi2(0) = Gamma_prod.at<double>(0,1);
        temp_xi2(1) = Gamma_prod.at<double>(1,1);
        temp_xi2(2) = Gamma_prod.at<double>(2,1);
        
        // orient xi1 along the measured flow direction, then keep Gamma a proper rotation (det = +1)
        double epsilon = 1e-6;
        double measDiffDot1 = temp_xi1.dot(tau_k);

        if(measDiffDot1<epsilon){
          Gamma_prod.col(0) = -Gamma_prod.col(0);
        }

        if(cv::determinant(Gamma_prod)<0.5){
          Gamma_prod.col(1)= -Gamma_prod.col(1);
        }

        Gamma = Gamma_prod;
        Lambda = Lambda_prod;

        est_beta_ = Lambda.at<double>(0,0);

        est_xi_1.x = Gamma.at<double>(0,0);
        est_xi_1.y = Gamma.at<double>(1,0);
        est_xi_1.z = Gamma.at<double>(2,0);
        est_xi_2.x = Gamma.at<double>(0,1);
        est_xi_2.y = Gamma.at<double>(1,1);
        est_xi_2.z = Gamma.at<double>(2,1);

        A = Gamma * Lambda * Gamma.t();

        cv::Mat R_incr = Gamma*Gamma_1.t();
        rot_incr_.push_back(R_incr);
        dt_vec_.push_back(dt);
        Gamma_1 = Gamma;
        noZcount=0;
    }
    }

      cv::Mat computeGamma(Point3d Z_s, double phi_s){
        cv::Mat Gamma_s = cv::Mat::zeros(3, 3, CV_64F);

        double theta = acos(min(max(Z_s.z,-1.0),1.0));
        cv::Mat R_mu = cv::Mat::eye(3, 3, CV_64F);
        double epsilon = 1e-6;

        if (std::abs(theta) > epsilon) {
            cv::Vec3d Zp(0.0, 0.0, 1.0);
            cv::Vec3d u = Zp.cross(Z_s);
            u = u / cv::norm(u);
            cv::Vec3d axis_angle = u * theta;
            cv::Rodrigues(axis_angle, R_mu);
        }

        cv::Mat R_phi = (cv::Mat_<double>(3,3) <<
            cos(phi_s), -sin(phi_s), 0,
            sin(phi_s),  cos(phi_s), 0,
            0,         0,        1
            );

      cv::Mat Gamma_meas = R_mu.t() * R_phi;

      cv::Point3d xi1_s(Gamma_meas.at<double>(0,0), Gamma_meas.at<double>(1,0), Gamma_meas.at<double>(2,0));
      cv::Point3d xi2_s(Gamma_meas.at<double>(0,1), Gamma_meas.at<double>(1,1), Gamma_meas.at<double>(2,1));

        Gamma_s = (cv::Mat_<double>(3,3) <<
          xi1_s.x,    xi2_s.x,  Z_s.x,    
          xi1_s.y,    xi2_s.y,  Z_s.y,
          xi1_s.z,    xi2_s.z,  Z_s.z
        );

        return Gamma_s;
  }

  /// @brief Builds Gamma = [xi1 xi2 Z_s] where xi1 is the flow point F_s projected on the tangent plane at Z_s.
  cv::Mat computeGammaFromFlow(Point3d Z_s, Point3d F_s){
        cv::Vec3d v_tangent = F_s - (Z_s).dot(F_s) * Z_s;

        if(cv::norm(v_tangent)<1e-6){
          std::cout << " Warning: flow vector too small, cannot compute Gamma from flow" << std::endl;
          return computeGamma(Z_s,0.0);
        }

        v_tangent = v_tangent / cv::norm(v_tangent);


        cv::Point3d xi_1(v_tangent[0], v_tangent[1], v_tangent[2]);

        cv::Vec3d v_xi2 = cv::Vec3d(Z_s.x, Z_s.y, Z_s.z).cross(v_tangent);
        v_xi2 = v_xi2 / cv::norm(v_xi2);

        cv::Point3d xi_2(v_xi2[0], v_xi2[1], v_xi2[2]);

        cv::Mat Gamma_s = (cv::Mat_<double>(3,3) <<
        xi_1.x,    xi_2.x,    Z_s.x,
        xi_1.y,    xi_2.y,    Z_s.y,
        xi_1.z,    xi_2.z ,   Z_s.z
        );

        return Gamma_s;

  }

  double getPhi(const cv::Mat& Gamma) {
    double epsilon = 1e-4;
    cv::Vec3d z_axis(0.0, 0.0, 1.0);
    cv::Vec3d mu(Gamma.at<double>(0,2),Gamma.at<double>(1,2),Gamma.at<double>(2,2));

    double theta_mu = acos(std::min(1.0, std::max(-1.0, z_axis.dot(mu))));

    cv::Mat Rg = cv::Mat::eye(3, 3, CV_64F);

    if (std::abs(theta_mu) < epsilon) {
        cv::Vec3d axis = cv::Vec3d(1.0, 0.0, 0.0) * CV_PI;
        cv::Rodrigues(axis, Rg);
    }
    else if (std::abs(theta_mu - CV_PI) < epsilon) {
        Rg = cv::Mat::eye(3, 3, CV_64F);
    }
    else {
        cv::Vec3d axis = z_axis.cross(mu);
        axis = axis / cv::norm(axis);
        cv::Vec3d axis_angle = axis * theta_mu;
        cv::Rodrigues(axis_angle, Rg);
    }

    cv::Mat Gamma_Np = Rg * Gamma;

    double phi = atan2(Gamma_Np.at<double>(1,0), Gamma_Np.at<double>(0,0));
    return phi;
  }
};

#endif // KENT_ESTIMATOR_H
