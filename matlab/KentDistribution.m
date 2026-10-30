classdef KentDistribution < AbstractHypersphericalDistribution
    % Represents a Kent distribution (or "balanced" Fisher-Bingham
    % distribution with 5-parameters).
    %
    % Kent, J. T., Hussein, I., & Jah, M. K. (2016, July). 
    % Directional distributions in tracking of space debris. 
    % In 2016 19th International Conference on Information Fusion (FUSION) 
    % (pp. 2081-2086). IEEE.
    %
    % Kent, J. T. (1982). 
    % The Fisher‐Bingham distribution on the sphere.
    % Journal of the Royal Statistical Society: Series B (Methodological), 
    % 44(1), 71-80.
    
    
    properties
        kappa (1,1) double  % concentration (scalar)
        beta (1,1) double   % ovalness (scalar)
        mu (:,1) double  % mode 
        xi1 (:,1) double % major axis
        xi2 (:,1) double % minor axis
        C {mustBeNonnegative} % normalization constant

        A (:,:) double % ovalness matrix
        Gamma (:,:) double % orthogonal basis matrix
        Lambda (:,:) double % matrix of eigein vector

        %Such as: C^{-1} exp( k mu^t x + x^t A x )
        %with 
        %A = Gamma Lambda Gamma^t 
        %where Lambda = diag(beta, -beta, 0)
        %      Gamma = [xi1, xi2, mu]
        %Gamma is orthogonal, tr(A) = 0. 

    end
    
    methods
        function Kent = KentDistribution(mu_, xi1_, xi2_, kappa_, beta_)
          % Constructor
          %
          % Parameters:
          %   mu_ (d x 1)
          %       mode (unit vector)
          %   xi1_ (d x 1)
          %       major axis (unit vector)
          %   xi2_ (d x 1)
          %       minor axis (unit vector)
          %   kappa_ (scalar)
          %       concentration parameter (>=0)
          %   beta_ (scalar)
          %       ovalness parameter (>=0)
            arguments
                mu_ (:,1) double 
                xi1_ (:,1) double 
                xi2_ (:,1) double 
                kappa_ (1,1) double {mustBeNonnegative}
                beta_ (1,1) double  {mustBeNonnegative}   
            end
            Kent.dim = size(mu_,1);
            assert(Kent.dim == 3, " This distribution is implemented at dimension 3 only (for now...).");

            epsilon = 1E-6;
            assert(2*beta_/kappa_ <1, "The distribution must be unimodal, ie, kappa must be > 2*beta, in its current implementation.");
            Gamma_ = [xi1_,xi2_,mu_];
            assert(abs(det(Gamma_))-1<epsilon, "Gamma = [xi1_,xi2_,mu_] must be orthogonal");
            Lambda_ = [beta_, 0,     0;
                       0,    -beta_, 0;
                       0      0,     0];
            A_ = Gamma_ * Lambda_ * Gamma_';
            assert(trace(A_)<epsilon, "Trace(Gamma * Lambda * Gamma') must be null, with Lambda = diag(beta_,-beta_, 0) and Gamma = [xi1_,xi2_,mu_]");
            
            Kent.kappa = kappa_;
            Kent.beta = beta_;
            Kent.mu = mu_;
            Kent.xi1 = xi1_;
            Kent.xi2 = xi2_;

            Kent.A = A_;
            Kent.Lambda = Lambda_;
            Kent.Gamma = Gamma_;

            %Compute C
            C_1=0;

            if not(Kent.beta == 0) 
                if Kent.kappa<50
                    for i = 0:100
                    C_1 = gamma(i+0.5)/gamma(i+1) * Kent.beta^(2*i)*(0.5*Kent.kappa)^(-2*i-0.5) * besselk(2*i+0.5,Kent.kappa) + C_1;
                    end
                    C_1 = 2*pi*C_1;
                else % asymptotic approximation for large Kappa
                    C_1 = 2*pi * exp(Kent.kappa)*(((Kent.kappa-2*Kent.beta)*(Kent.kappa+2*Kent.beta))^(-0.5));
                end
            else % back to a VMF distribution
            C_1 = 4*pi * sinh(Kent.kappa)/Kent.kappa;
            end
            Kent.C = 1/C_1; 
        end
        
        function p = pdf(this, xa)
            % Evaluates pdf at each column of xa
            % Parameters:
            %   xa (d x n matrix)
            %       each column represents one of the n points in R^d that the
            %       pdf is evaluated at; can be just a (d x 1) vector as well
            % Returns:
            %   p (1 x n row vector)
            %       values of the pdf at each column of xa
            arguments
                this (1,1) KentDistribution
                xa (:,:) double
            end
            assert (size(xa,1) == size(this.mu,1));

            p = this.C * exp(this.kappa * this.mu' * xa + xa' * this.A * xa);
        end

        function mu = meanDirection(this)
            arguments
                this (1,1) KentDistribution
            end
           mu = this.mu;
        end

        function psi = headingAngle(this)
            arguments
                this (1,1) KentDistribution
            end
        
            epsilon = 1e-4;
            z_axis = [0; 0; 1];
            mu = this.mu;
        
          
            theta_mu = acos(dot(z_axis, mu));  
        
          
            if abs(theta_mu) < epsilon
                 Rg = rotationVectorToMatrix(pi * [1; 0; 0]);
        
           
            elseif abs(theta_mu - pi) < epsilon
               
                Rg = eye(3);
        
       
            else
                axe_mu = cross(z_axis, mu);
                axe_mu = axe_mu / norm(axe_mu);
                Rg = rotationVectorToMatrix(theta_mu * axe_mu);
            end
        
          
            Gamma_Np = Rg * this.Gamma;
        
            
            psi = atan2(Gamma_Np(2,1), Gamma_Np(1,1));
        end

        function s = sampleTerdik(this, n)
            % Probabilistic sampling based on
            % gaussian envelope.
            % Returns:
            %   samples (d x n matrix)
            %       one sample per column
            %
            % Jammalamadaka, S. R., & Terdik, G. (2022). 
            % Simulation and visualization of 3d-spherical distributions. 
            % In Directional Statistics for Innovative Applications: A Bicentennial Tribute to Florence Nightingale 
            % (pp. 119-145). Singapore: Springer Nature Singapore.
            arguments
                this (1,1) KentDistribution
                n (1,1) double % size of the sampling
            end
            % sampleTerdik is written with a (mu,psi) parameterization,
            % with psi the z-axis rotation from the (x1,x2) reference basis
            % to (xi1,xi2) basis of the ellipse. In order to use this
            % function, we thus have to estimate this parameter psi from
            % the orthogonal matrix Gamma.

            psi = this.headingAngle();

            % now we can obtain the sample

            s = Random_FB5_Kent(this.kappa,this.beta,this.mu,psi,n);
            
          end

          function s = sampleMardiaKent(this, n)
                % Probabilistic sampling based on angular gaussian envelope.
                % Returns:
                %   samples (d x n matrix)
                %       one sample per column
                %
                % Kent, J. T., Ganeiber, A. M., & Mardia, K. V. (2018). 
                % A new unified approach for the simulation of a wide class
                % of directional distributions. 
                % Journal of Computational and Graphical Statistics,
                % 27(2), 291-301.

                arguments
                    this (1,1) KentDistribution
                    n (1,1) double % size of the sampling
                end

            s = zeros(this.dim, n);
            i = 1;
            Astar = this.A + this.kappa/2 * (eye(this.dim) - this.mu * this.mu');
            Aminus = - Astar; % Kent uses a minus sign here!
            q = this.dim;

            % compute b
            [G,L] = eig(Astar);
            bfun = @(b) sum(1./(b-2*L)) - 1;
            b = fsolve(bfun, 1, optimset('display', 'none'));
            Omega = eye(this.dim) + 2*Astar/b;
            %
            %
            Mbstarkappa = exp(-(q-b)/2) * (q/b)^(q/2)* exp(this.kappa);
            nReject = 0;
            ffishbingstar = @(x) exp(this.kappa * this.mu' * x + x' * this.A * x);
            %ffishbingstar = @(x) exp(-x' * Aminus * x);
            facgstar = @(x) (x' * Omega * x)^(-q/2);
            inv(Omega)
            while(i<=n)
            % draw x from angular central Gaussian
                y = mvnrnd(zeros(this.dim,1), inv(Omega))';
                x = y/norm(y);
                % check rejection
                W = rand(1);
                if W < ffishbingstar(x) /(Mbstarkappa * facgstar(x))
                    s(:,i) = x;
                    i = i + 1;
                else
                    nReject = nReject + 1;
                end   
            end
       
          end

          function K = multiplySolver(this, K2)
            % Computes the product of two Kent pdfs
            %
            % Parameters:
            %   K2 (KentDistribution)
            %       second Kent Distribution
            % Returns:
            %   K (KentDistribution)
            %       Kent distribution representing this*K2 (after
            %       renormalization)
            arguments
                this (1,1) KentDistribution
                K2 (1,1) KentDistribution
            end
            assert(isa(K2, 'KentDistribution'));

            mu_m = this.kappa*this.mu + K2.kappa*K2.mu;
            kappa_m = norm(mu_m);
            mu_m = mu_m /kappa_m;

            A_sum = this.A + K2.A;

            Afun = @(M) norm(M-A_sum,'fro');

            function [c,ceq] = Aconstr(C)
               
                c = [];
            
                ceq(1) = trace(C);
            
                sym = C - C';
                ceq(2) = sym(1,1);
                ceq(3) = sym(1,2);
                ceq(4) = sym(1,3);
                ceq(5) = sym(2,1);
                ceq(6) = sym(2,2);
                ceq(7) = sym(2,3);
                ceq(8) = sym(3,1);
                ceq(9) = sym(3,2);
                ceq(10) = sym(3,3);
            
                mode = C*mu_m;
                ceq(11) = mode(1,1);
                ceq(12) = mode(2,1);
                ceq(13) = mode(3,1);
            end

            nonlcon = @Aconstr;
            A0=eye(3);
            B = [];
            b = [];
            Beq = [];
            beq = [];
            lb = -inf;
            ub = inf;

            Aopt = fmincon(Afun, A0, B, b, Beq, beq, lb, ub,  nonlcon);

            [Gamma_m,Lambda_m] = eig(Aopt); % Eigenvalue decomposition
            [Lambda_m, order] = sort(diag(Lambda_m),'descend');  % sort eigenvalues
            Gamma_m = Gamma_m(:,order);

            temp_g = Gamma_m(:,2);
            Gamma_m(:,2)=Gamma_m(:,3);
            Gamma_m(:,3)=temp_g;

            % temp_l = Lambda_m(2);
            % Lambda_m(2)=Lambda_m(3);
            % Lambda_m(3)=temp_l;

            beta_m = norm(Aopt);

            xi1_m = Gamma_m(:,1);
            xi2_m = Gamma_m(:,2);

            K = KentDistribution(mu_m,xi1_m,xi2_m,kappa_m,beta_m);
      end  

  function K = multiply(this, K2)
            % Computes the product of two Kent pdfs
            %
            % Parameters:
            %   K2 (KentDistribution)
            %       second Kent Distribution
            % Returns:
            %   K (KentDistribution)
            %       Kent distribution representing this*K2 (after
            %       renormalization)
            arguments
                this (1,1) KentDistribution
                K2 (1,1) KentDistribution
            end
            assert(isa(K2, 'KentDistribution'));
            
            p = this.dim;
            mu_m = this.kappa*this.mu + K2.kappa*K2.mu;
            kappa_m = norm(mu_m);
            mu_m = mu_m /kappa_m;

            A_sum = this.A + K2.A;
            
            Aopt = A_sum*(eye(p,p)-mu_m*mu_m') + 1/(p-1)*mu_m'*A_sum*mu_m*(eye(p,p)+(p-2)*mu_m*mu_m')-mu_m*mu_m'*A_sum;%unique solution

            [Gamma_m,Lambda_m] = eig(Aopt); % Eigenvalue decomposition --> possible switch since non-null eigenvalues are opposed
            [Lambda_m, order] = sort(diag(Lambda_m),'descend');  % sort eigenvalues
            Gamma_m = Gamma_m(:,order);

            temp_g = Gamma_m(:,2);
            Gamma_m(:,2)=Gamma_m(:,3);
            Gamma_m(:,3)=temp_g;

            % temp_l = Lambda_m(2);
            % Lambda_m(2)=Lambda_m(3);
            % Lambda_m(3)=temp_l;

            beta_m = norm(Aopt);

            xi1_m = Gamma_m(:,1);
            xi2_m = Gamma_m(:,2);

            %phi_m_originial = this.headingAngle()
            %phi_m_noise = K2.headingAngle()

            K = KentDistribution(mu_m,xi1_m,xi2_m,kappa_m,beta_m);
            %phi_m_resultat = K.headingAngle()
  end  

  function K = multiplyArbitrary(this, K2)
            % Computes the product of two Kent pdfs with memory of last
            % iteration
            %
            % Parameters:
            %   K2 (KentDistribution)
            %       second Kent Distribution
            % Returns:
            %   K (KentDistribution)
            %       Kent distribution representing this*K2 (after
            %       renormalization)
            arguments
                this (1,1) KentDistribution
                K2 (1,1) KentDistribution
            end
            assert(isa(K2, 'KentDistribution'));
            xi1_1 = this.xi1;

            p = this.dim;
            mu_m = this.kappa*this.mu + K2.kappa*K2.mu;
            kappa_m = norm(mu_m);
            mu_m = mu_m /kappa_m;

            A_sum = this.A + K2.A;
            
            Aopt = A_sum*(eye(p,p)-mu_m*mu_m') + 1/(p-1)*mu_m'*A_sum*mu_m*(eye(p,p)+(p-2)*mu_m*mu_m')-mu_m*mu_m'*A_sum;%unique solution

            [Gamma_m,Lambda_m] = eig(Aopt); % Eigenvalue decomposition --> possible switch since non-null eigenvalues are opposed
            [Lambda_m, order] = sort(diag(Lambda_m),'descend');  % sort eigenvalues
            Gamma_m = Gamma_m(:,order);

            temp_g = Gamma_m(:,2);
            Gamma_m(:,2)=Gamma_m(:,3);
            Gamma_m(:,3)=temp_g;

            % temp_l = Lambda_m(2);
            % Lambda_m(2)=Lambda_m(3);
            % Lambda_m(3)=temp_l;

            beta_m = norm(Aopt);

            xi1_m = Gamma_m(:,1);
            xi2_m = Gamma_m(:,2);
            epsilon = 1e-4;
            if dot(xi1_m,K2.xi1)<epsilon
                xi1_m = -xi1_m;
                xi2_m = -xi2_m;
            end

            Gamma_temp = [xi1_m,xi2_m,mu_m];

            if det(Gamma_temp)<0.5
                xi2_m = -xi2_m;
            end


            %phi_m_originial = this.headingAngle()
            %phi_m_noise = K2.headingAngle()

            K = KentDistribution(mu_m,xi1_m,xi2_m,kappa_m,beta_m);
            %phi_m_resultat = K.headingAngle()
  end  

  function K = convolve(this, K2)
            % Computes the convolution of two aligned Kent pdfs
            % Note that this operation is not a real convolution, it
            % produces a Kent distribution of equal orthogonal basis
            % [xi_1,xi_2,mu] with updated parameters kappa and beta.
            % If the two distributions have different orthogonal basis,
            % only the first one is kept.
            %
            % Parameters:
            %   K2 (KentDistribution)
            %       second Kent Distribution
            % Returns:
            %   K (KentDistribution)
            %       Kent distribution representing this (*) K2 (after
            %       renormalization)
            arguments
                this (1,1) KentDistribution
                K2 (1,1) KentDistribution
            end
            assert(isa(K2, 'KentDistribution'));

            mu_1 = this.mu;
            xi1_1 = this.xi1;
            xi2_1 = this.xi2;

            kappa_1=this.kappa;
            beta_1=this.beta;
            kappa_2=K2.kappa;
            beta_2=K2.beta;

            % Computations of moments
            R_1 = 1 - kappa_1/((kappa_1-2*beta_1)*(kappa_1+2*beta_1));
            Q_1 = 4*beta_1/((kappa_1-2*beta_1)*(kappa_1+2*beta_1));
            R_2 = 1 - kappa_2/((kappa_2-2*beta_2)*(kappa_2+2*beta_2));
            Q_2 = 4*beta_2/((kappa_2-2*beta_2)*(kappa_2+2*beta_2));

            % Convolution
            R_conv = R_1*R_2;
            Q_conv = Q_1 + Q_2;

            % Parameter estimation
            kappa_conv = 4*(1-R_conv)/(4*(1-R_conv)^2 - Q_conv^2);
            beta_conv = Q_conv/(4*(1-R_conv)^2 - Q_conv^2);

            K = KentDistribution(mu_1,xi1_1,xi2_1,kappa_conv,beta_conv);
    end  

    end
end











%%
% A_m = this.A + K2.A; % new exponent
% 
%                 A_m = 0.5*(A_m+A_m'); % Ensure symmetry of A, asymmetry may arise as a consequence of a numerical instability earlier.
% 
%                 [Gamma_m,Lambda_m] = eig(A_m); % Eigenvalue decomposition
%                 [Lambda_m, order] = sort(diag(D),'descend');  % sort eigenvalues
%                 Gamma_m = Gamma_m(:,order);
%                 Lambda_m = Lambda_m - Lambda_m(end); % last entry should be zero
% 
%                 A_m = Gamma_m * Lambda_m * Gamma_m';
% 
% 
% 
%                 xi1_m = Gamma_m(:,1);
%                 xi2_m = Gamma_m(:,2);
%%