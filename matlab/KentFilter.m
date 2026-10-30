classdef KentFilter < AbstractHypersphericalFilter
    % Filter based von the Kent distribution.
    %

    properties
        state (1,1) KentDistribution = KentDistribution([0;0;1],[1;0;0],[0;1;0] , 100, 5);
    end
        
    methods 
        function setState(this, state_)
            assert(isa(state_, 'KentDistribution'));
            this.state = state_;
        end
        
        function est = getEstimate(this)
           est = this.state; 
        end
        
        function predictIdentity(this, sysNoise)
            % Predicts assuming identity system model, i.e.,
            % f(x(k+1)|x_k) = Kent(x(k+1) ; mu = mu_k, kappa_k^w)
            % where w is noise given by sysNoise.
            %
            % Parameters:
            %   sysNoise (KentDistribution)
            %       distribution of noise
            assert(isa(sysNoise, 'KentDistribution'));
            
            this.state = this.state.convolve(sysNoise);
        end
        
        function predictLinear(this, R_k, sysNoise)
            % Predicts assuming adaptive system model R_k in SO(3), i.e.,
            % f(x(k+1)|x_k) = Kent(x(k+1) ; [xi1, xi2, mu] = R_k*[xi1_k, xi2_k, mu_k], kappa_k^w, beta_k^w)
            % where w is noise given by sysNoise.
            %
            % Parameters:
            %   R_k (3x3 matrix)
            %       Rotation matrix representing system model
            %   sysNoise (KentDistribution)
            %       distribution of noise            
            assert(isa(sysNoise, 'KentDistribution'));
            %assert(ismatrix(R_k)&&size(R_k,1)==size(R_k,2)&&size(R_k,2)==3&&R_k==transpose(R_k)&&det(R_k)==1, "R_k is not a rotation matrix");
            
            Gamma_k = [this.state.xi1,this.state.xi2,this.state.mu];
            
            Gamma_p = R_k * Gamma_k;
   
            this.state = KentDistribution(Gamma_p(:,3),Gamma_p(:,1),Gamma_p(:,2),this.state.kappa, this.state.beta);
            this.state = this.state.convolve(sysNoise);
        end
        
       
        
        function updateIdentity(this, measNoise, z, psi)
            % Updates assuming identity measurement model, i.e.,
            % f(z(k)|x(k=) = Kent(z(k),psi(k); [xi1_k^psi, xi2_k^psi, mu_k^z] , kappa_k^v, beta_k^v)
            % where v is noise given by measNoise.
            %
            % Parameters:
            %   measNoise (KentDistribution)
            %       distribution of noise
            %   z (vector)
            %       measurement in R^3      
            %   psi (scalar)
            %       measurement in [0,2pi)
            
            
            assert(isa(measNoise, 'KentDistribution'));
            assert(measNoise.mu(end)==1,'Set mu of measNoise to [0,0,...,1] to acknowledge that the mean is discarded');
            assert(size(z,1) == this.dim);
            assert(size(z,2) == 1);

            % We update mu, xi1 & xi2 of the noise
            
            R_psi = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
            

            Zp=[0,0,1];
            z=z/norm(z);
            theta=acos(z(3));
            R_mu = eye(3);
            epsilon = 1e-6;
            if abs(theta)>epsilon
            u=cross(Zp,z);%axis of rotation
            u=u/norm(u);
            R_mu = rotationVectorToMatrix(u*theta);
            end
            
            Gamma_meas = R_mu' * R_psi;

            xi1_u=Gamma_meas(:,1);
            xi2_u=Gamma_meas(:,2);

            measNoiseUpdated = KentDistribution(z,xi1_u,xi2_u,measNoise.kappa,measNoise.beta);

            %psi_original=psi
            %psi_extraite = measNoiseUpdated.headingAngle()
            %mu_original = z
            %mu_extrait = Gamma_meas(:,3)
            
            %phi_state_before=this.state.headingAngle()
            this.state = this.state.multiply(measNoiseUpdated);
            %phi_state_after=this.state.headingAngle()
        end

         function updateIdentityArbitrary(this, measNoise, z, psi)
            % Updates assuming identity measurement model, i.e.,
            % f(z(k)|x(k=) = Kent(z(k),psi(k); [xi1_k^psi, xi2_k^psi, mu_k^z] , kappa_k^v, beta_k^v)
            % where v is noise given by measNoise.
            %
            % Parameters:
            %   measNoise (KentDistribution)
            %       distribution of noise
            %   z (vector)
            %       measurement in R^3      
            %   psi (scalar)
            %       measurement in [0,2pi)
            
            
            assert(isa(measNoise, 'KentDistribution'));
            assert(measNoise.mu(end)==1,'Set mu of measNoise to [0,0,...,1] to acknowledge that the mean is discarded');
            assert(size(z,1) == this.dim);
            assert(size(z,2) == 1);

            % We update mu, xi1 & xi2 of the noise
            
            R_psi = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
            

            Zp=[0,0,1];
            z=z/norm(z);
            theta=acos(z(3));
            R_mu = eye(3);
            epsilon = 1e-6;
            if abs(theta)>epsilon
            u=cross(Zp,z);%axis of rotation
            u=u/norm(u);
            R_mu = rotationVectorToMatrix(u*theta);
            end
            
            Gamma_meas = R_mu' * R_psi;

            xi1_u=Gamma_meas(:,1);
            xi2_u=Gamma_meas(:,2);

            measNoiseUpdated = KentDistribution(z,xi1_u,xi2_u,measNoise.kappa,measNoise.beta);

            %psi_original=psi
            %psi_extraite = measNoiseUpdated.headingAngle()
            %mu_original = z
            %mu_extrait = Gamma_meas(:,3)
            
            %phi_state_before=this.state.headingAngle()
            this.state = this.state.multiplyArbitrary(measNoiseUpdated);
            %phi_state_after=this.state.headingAngle()
        end

        function updateIdentityFromHeading(this, measNoise, z, h)
            % Updates assuming identity measurement model & that the measurement includes the heading vector, i.e.,
            % f(z(k)|x(k=) = Kent(z(k),psi(k); [xi1_k^psi, xi2_k^psi, mu_k^z] , kappa_k^v, beta_k^v)
            % where v is noise given by measNoise.
            %
            % Parameters:
            %   measNoise (KentDistribution)
            %       distribution of noise
            %   z (vector)
            %       measurement in R^3      
            %   h (vector)
            %       measurement in R^3
            
            
            assert(isa(measNoise, 'KentDistribution'));
            assert(measNoise.mu(end)==1,'Set mu of measNoise to [0,0,...,1] to acknowledge that the mean is discarded');
            assert(size(z,1) == this.dim);
            assert(size(z,2) == 1);

            % We update mu, xi1 & xi2 of the noise
            
            xi2_u=cross(z,h);

            measNoiseUpdated = KentDistribution(z,h,xi2_u,measNoise.kappa,measNoise.beta);

            %psi_original=psi
            %psi_extraite = measNoiseUpdated.headingAngle()
            %mu_original = z
            %mu_extrait = Gamma_meas(:,3)
            
            %phi_state_before=this.state.headingAngle()
            this.state = this.state.multiplyArbitrary(measNoiseUpdated);
            %phi_state_after=this.state.headingAngle()
        end


        function mean = getEstimateMean(this)
            mean = this.state.mu;
        end

        function heading = getEstimateHeading(this)
            heading = this.state.headingAngle();
        end
    end
end
