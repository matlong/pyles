"""PyTorch implementation of 3D Navier-Stokes equations.
   Fourier method in horizontal direction and Chebyshev method in vertical direction. 
   Runge-Kutta 3rd order time stepping.
   Copyright 2024 Long Li, ODYSSEY Team INRIA Rennes.
"""
import numpy as np
import torch

def cheby_grid(nx, **arr_kwargs):
    """Chebyshev nodes, differentiation matrix and quadrature coefficients on [1,-1]."""
    # Chebyshev-Lobatto points
    theta = torch.linspace(0, torch.pi, nx, **arr_kwargs).reshape(-1,1)
    x = torch.cos(theta)
    
    # Diff. matrix
    c = torch.ones_like(x)
    c[[0,-1]] *= 2.
    c *= (-1)**torch.arange(nx, **arr_kwargs).reshape(-1,1)
    X = x.tile(1,nx)
    dX = X - X.T
    D = c @ (1./c).T / (dX + torch.eye(nx, **arr_kwargs))
    D -= torch.diag(D.sum(dim=-1)) 
    
    # Quadrature coef. (weight for integration) 
    W = torch.zeros_like(x)
    v = torch.ones((nx-2,1), **arr_kwargs)
    n = nx - 1
    if n%2 == 0:
        W[0] = 1/(n*n-1)
        W[-1] = 1/(n*n-1)
        for k in range(1,n//2):
            v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
        v = v - torch.cos(n*theta[1:-1])/(n*n-1)
    else:
        W[0] = 1/(n*n)
        W[-1] = 1/(n*n)
        for k in range(1,(n-1)//2+1):
            v = v - 2*torch.cos(2*k*theta[1:-1])/(4*k*k-1)
    W[1:-1] = 2*v/n
    return x.squeeze(), D, W.squeeze()


class NS3D:

    def __init__(self, param):       
        self.ne = param['ne']
        self.nz = param['nz']
        self.ny = param['ny']
        self.nx = param['nx']
        self.H = param['H']
        self.delta = param['delta']
        self.Ly = param['Ly']
        self.Lx = param['Lx']
        self.f = param['f']
        self.Cs = param['Cs']
        self.num = param['num']
        self.cfl = param['cfl']
        self.device = param['device']
        self.dtype = param['dtype']
        self.arr_kwargs = {'dtype':self.dtype, 'device':self.device} 

        self.set_model()
        self.base_shape = (self.ne,self.nz,self.ny,self.nx)
        self.u = torch.zeros(self.base_shape, **self.arr_kwargs)
        self.v = torch.zeros(self.base_shape, **self.arr_kwargs)
        self.w = torch.zeros(self.base_shape, **self.arr_kwargs)
        # [WIP] move pressure to state variables (self.p)?
      
        # Wind stress
        tau_shape = (1,self.ny,self.nx)
        taux = param['tau_mean'][0] * torch.ones(tau_shape, **self.arr_kwargs)
        tauy = param['tau_mean'][1] * torch.ones(tau_shape, **self.arr_kwargs)
        self.taux_hat, self.tauy_hat = torch.fft.fft2(taux), torch.fft.fft2(tauy)


    def set_model(self):
        """Set model's grids and operators."""
        # Horizontal physical grid
        self.dx, self.dy = self.Lx/self.nx, self.Ly/self.ny
        self.x = torch.linspace(self.dx/2, self.Lx-self.dx/2, self.nx, **self.arr_kwargs)
        self.y = torch.linspace(self.dy/2, self.Ly-self.dy/2, self.ny, **self.arr_kwargs)
        
        # Horizontal spectral grid 
        kx = torch.fft.fftfreq(self.nx, self.dx/(2*np.pi), **self.arr_kwargs)
        ky = torch.fft.fftfreq(self.ny, self.dy/(2*np.pi), **self.arr_kwargs)
        self.kx, self.ky = torch.meshgrid(kx, ky, indexing='xy')
        self.k2 = self.kx**2 + self.ky**2

        # 2/3 anti-aliasing mask
        maskx = (abs(self.kx) < (2/3)*abs(self.kx).max())
        masky = (abs(self.ky) < (2/3)*abs(self.ky).max()) 
        self.mask = maskx * masky

        # Vertical Chebyshev grid, gradient and integration operators
        self.z, self.Dz, self.Wz = cheby_grid(self.nz, **self.arr_kwargs) # on [1,-1]
        a, b = 0.5*(self.delta - self.H), 0.5*(self.H + self.delta)
        self.z = a*self.z + b  # on [delta,H]
        self.Dz, self.Wz = self.Dz/a, abs(a)*self.Wz
        Dz2 = self.Dz @ self.Dz # 2nd derivative

        # Spectral inverse Laplacian operator
        self.L_inv = torch.zeros((self.nz,self.nz,self.ny,self.nx), **self.arr_kwargs)
        for i in range(self.ny):
            for j in range(self.nx):
                L = Dz2 - self.k2[i,j] * torch.eye(self.nz, **self.arr_kwargs)
                # homogeneous Neumman condition for pressure
                L[0], L[-1] = self.Dz[0], self.Dz[-1] 
                self.L_inv[:,:,i,j] = torch.linalg.inv(L)
        
        # Convert to complex type to match fft
        self.Dz = torch.complex(self.Dz, torch.zeros_like(self.Dz)) 
        self.L_inv = torch.complex(self.L_inv, torch.zeros_like(self.L_inv))

        # Constant Smagorinsky coefficient
        delta = (self.dx * self.dy * abs(torch.diff(self.z)))**(1/3)
        delta = torch.cat((delta, delta[-1:]))
        self.smag_coef = (self.Cs * delta.view(self.nz,1,1))**2

        # CFL constant for timestep
        dz_min = abs(torch.diff(self.z)).min()
        self.cfl_delta_min = self.cfl * min(self.dx, self.dy, dz_min)


    def compute_timestep(self):
        """Estimate time step by a fixed CFL number."""
        umax = max(abs(self.u).max(), abs(self.v).max(), abs(self.w).max())
        run.dt = self.cfl_delta_min / umax


    def Smagorinsky_SGS(self, u_hat, v_hat, w_hat):
        """Compute the Smagorinsky SGS tensor `tau`."""
        # diagonal components of strain rate tensor
        Sxx = torch.fft.ifft2(1j*self.kx * u_hat).real
        Syy = torch.fft.ifft2(1j*self.ky * v_hat).real
        Szz = torch.fft.ifft2(torch.einsum('kl,rlji->rkji', self.Dz, w_hat)).real
        
        # off-diagonal components of strain rate tensor
        Sxy = 0.5*torch.fft.ifft2(1j*self.kx*v_hat + 1j*self.ky*u_hat).real
        Sxz = 0.5*torch.fft.ifft2(1j*self.kx*w_hat + torch.einsum('kl,rlji->rkji',self.Dz,u_hat)).real
        Syz = 0.5*torch.fft.ifft2(1j*self.ky*w_hat + torch.einsum('kl,rlji->rkji',self.Dz,v_hat)).real 

        # eddy viscosity
        nu = self.num + self.smag_coef * torch.sqrt(2*(Sxx**2 + Syy**2 + Szz**2 + 2*Sxy**2+ 2*Sxz**2 + 2*Syz**2))
        # [WIP] move to dynamic Smagorinsky? 

        # FFT2 of turbulent stress components
        tauxx_hat = torch.fft.fft2(2*nu * Sxx)
        tauyy_hat = torch.fft.fft2(2*nu * Syy)
        tauzz_hat = torch.fft.fft2(2*nu * Szz)
        tauxy_hat = torch.fft.fft2(2*nu * Sxy)
        tauxz_hat = torch.fft.fft2(2*nu * Sxz)
        tauyz_hat = torch.fft.fft2(2*nu * Syz)
        return tauxx_hat, tauyy_hat, tauzz_hat, tauxy_hat, tauxz_hat, tauyz_hat


    def compute_time_derivatives(self):
        """Compute RHS of 3D incompressible NS equations."""  
        # Nonlinear advection
        uu_hat = torch.fft.fft2(self.u * self.u) 
        vv_hat = torch.fft.fft2(self.v * self.v) 
        ww_hat = torch.fft.fft2(self.w * self.w) 
        uv_hat = torch.fft.fft2(self.u * self.v) 
        uw_hat = torch.fft.fft2(self.u * self.w) 
        vw_hat = torch.fft.fft2(self.v * self.w) 
        du_hat = - 1j*self.kx*self.mask*uu_hat - 1j*self.ky*self.mask*uv_hat - torch.einsum('kl,rlji->rkji',self.Dz,uw_hat)
        dv_hat = - 1j*self.kx*self.mask*uv_hat - 1j*self.ky*self.mask*vv_hat - torch.einsum('kl,rlji->rkji',self.Dz,vw_hat)
        dw_hat = - 1j*self.kx*self.mask*uw_hat - 1j*self.ky*self.mask*vw_hat - torch.einsum('kl,rlji->rkji',self.Dz,ww_hat)
        # [WIP] is it necessary to include anti-aliasing in vertical?

        # Coriolis force
        u_hat = torch.fft.fft2(self.u)
        v_hat = torch.fft.fft2(self.v)
        w_hat = torch.fft.fft2(self.w) 
        du_hat += self.f * v_hat 
        dv_hat -= self.f * u_hat
        # [WIP] buoyancy/gravity force in vertical?

        # Subgrid scales terms
        tauxx_hat, tauyy_hat, tauzz_hat, tauxy_hat, tauxz_hat, tauyz_hat = self.Smagorinsky_SGS(u_hat, v_hat, w_hat)
        tauxz_hat[:,0], tauyz_hat[:,0] = self.taux_hat, self.tauy_hat # surface momentum flux <- wind stress
        tauxz_hat[:,-1] *= 0. # zero bottom flux
        tauyz_hat[:,-1] *= 0.   
        du_hat += 1j*self.kx*tauxx_hat + 1j*self.ky*tauxy_hat + torch.einsum('kl,rlji->rkji',self.Dz,tauxz_hat) 
        dv_hat += 1j*self.kx*tauxy_hat + 1j*self.ky*tauyy_hat + torch.einsum('kl,rlji->rkji',self.Dz,tauyz_hat) 
        dw_hat += 1j*self.kx*tauxz_hat + 1j*self.ky*tauyz_hat + torch.einsum('kl,rlji->rkji',self.Dz,tauzz_hat) 

        # Free-slip vertical conditions
        dw_hat[:,0] *= 0.
        dw_hat[:,-1] *= 0.

        # Pressure correction
        rhs = 1j*self.kx*du_hat + 1j*self.ky*dv_hat + torch.einsum('kl,rlji->rkji',self.Dz,dw_hat) 
        rhs[:,0] *= 0. # homogeneous Neumann BC
        rhs[:,-1] *= 0. 
        dp_hat = torch.einsum('klji,rlji->rkji', self.L_inv, rhs) # invert Poisson equation
        du_hat -= 1j*self.kx * dp_hat 
        dv_hat -= 1j*self.ky * dp_hat
        dw_hat -= torch.einsum('kl,rlji->rkji', self.Dz, dp_hat)

        # Return tendencies in physical space
        du = torch.fft.ifft2(du_hat).real
        dv = torch.fft.ifft2(dv_hat).real
        dw = torch.fft.ifft2(dw_hat).real
        return du, dv, dw


    def step(self):
        """Time itegration with SSPRK3 scheme."""  
        dt0_u, dt0_v, dt0_w = self.compute_time_derivatives()
        self.u = self.u + self.dt * dt0_u
        self.v = self.v + self.dt * dt0_v
        self.w = self.w + self.dt * dt0_w

        dt1_u, dt1_v, dt1_w = self.compute_time_derivatives()
        self.u = self.u + (self.dt/4) * (dt1_u - 3*dt0_u)
        self.v = self.v + (self.dt/4) * (dt1_v - 3*dt0_v)
        self.w = self.w + (self.dt/4) * (dt1_w - 3*dt0_w)

        dt2_u, dt2_v, dt2_w = self.compute_time_derivatives()
        self.u = self.u + (self.dt/12) * (8*dt2_u - dt1_u - dt0_u)
        self.v = self.v + (self.dt/12) * (8*dt2_v - dt1_v - dt0_v)
        self.w = self.w + (self.dt/12) * (8*dt2_w - dt1_w - dt0_w)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    param = {
        'ne': 1, # ensemble size
        'nx': 128, # zonal grid size
        'ny': 128, # meridional grid size
        'nz': 128, # vertical grid size
        'Lx': 128., # zonal domain length (m)
        'Ly': 128., # merional domain length (m)
        'H': -64., # vertical domain depth (m)
        'delta': -0.25, # surface position (m)
        'f': 1e-4, # Coriolis frequency (s^-1)
        'tau_mean': [3.72e-5, 0.], # mean wind stress (m^2/s^2)
        'num': 1e-6, # molecular viscosity (m^2/s)
        'Cs': 0.2, # Smagorinsky constant (nondim)
        'dt': 60., # timestep (s)
        'cfl': 0.5, # fixed CFL number
        'dtype': torch.float64, # torch.float32 or torch.float64
        'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
    }
    run = NS3D(param)
 
    # Read Langmuir turbulence simulation data from Oceananignans
    data = np.load(f'spinup_{run.nz}x{run.ny}x{run.nx}.npz')
    run.u = torch.from_numpy(data['u']).to(run.device).type(run.dtype).tile(run.ne,1,1,1)
    run.v = torch.from_numpy(data['v']).to(run.device).type(run.dtype).tile(run.ne,1,1,1)
    data.close()

    # Reconstruct vertical velocity from continuity equation
    divh_hat = 1j*run.kx*torch.fft.fft2(run.u) + 1j*run.ky*torch.fft.fft2(run.v) # horizontal divergence
    rhs = torch.fft.ifft2(torch.einsum('kl,rlji->rkji', run.Dz, -divh_hat)).real
    Dz2 = run.Dz.real @ run.Dz.real
    Dz2_inv = torch.linalg.inv(Dz2[1:-1,1:-1]) # free_slip BC accounted
    run.w[:,1:-1] = torch.einsum('kl,rlji->rkji', Dz2_inv, rhs[:,1:-1])
    run.compute_timestep()

    # Time and control params
    t = 0. # start of simulation time (s)
    t_end = 86400. # 1 day
    step_checknan = 1. # (s)
    step_log = 1. # every min 
    step_plot = 0
    step_save = 5*60. # every 5 mins 
    outdir = '/srv/storage/ithaca@storage2.rennes.grid5000.fr/lli/LES'
    
    # Init output
    if step_save > 0:
        import os
        os.makedirs(outdir) if not os.path.isdir(outdir) else None
        filename = os.path.join(outdir, 'param.pth')
        torch.save(param, filename)
        filename = os.path.join(outdir, 'uvw_0.npz')
        np.savez(filename, t=t, 
                u=run.u.cpu().numpy().astype('float32'), 
                v=run.v.cpu().numpy().astype('float32'),
                w=run.w.cpu().numpy().astype('float32'))
        count_save = 1
        next_save = step_save 
    
    # Init figures
    if step_plot > 0:
        import matplotlib.pyplot as plt
        plt.ion()
        im_kwargs = {'cmap':'RdBu_r', 'origin':'lower', 'animated':True}
        f,a = plt.subplots(1,3)
        a[0].set_title('surface $u$')
        a[1].set_title('surface $v$')
        a[2].set_title('surface $w$')
        u = (run.u[0,1]).cpu().numpy()
        v = (run.v[0,1]).cpu().numpy()
        w = (run.w[0,1]).cpu().numpy()
        um, vm, wm = abs(u).max(), abs(v).max(), abs(w).max()
        a[0].imshow(u, vmin=-um, vmax=um, **im_kwargs)
        a[1].imshow(v, vmin=-vm, vmax=vm, **im_kwargs)
        a[2].imshow(w, vmin=-wm, vmax=wm, **im_kwargs)
        plt.tight_layout()
        plt.pause(0.1)
        next_plot = step_plot

    next_checknan = step_checknan
    next_log = step_log
    
    # Time-stepping
    while t < t_end:
        
        run.step()

        if step_checknan > 0 and t >= next_checknan: 
            if torch.isnan(run.u).any():
                raise ValueError('Stopping, NAN number in `u` at iteration {n}.')
            if torch.isnan(run.v).any():
                raise ValueError('Stopping, NAN number in `v` at iteration {n}.')
            if torch.isnan(run.w).any():
                raise ValueError('Stopping, NAN number in `w` at iteration {n}.')
            next_checknan += step_checknan

        if step_log > 0 and t >= next_log:
            t_, dt = t.cpu().numpy(), run.dt.cpu().numpy()
            ke = (run.Wz * (run.u**2 + run.v**2 + run.w**2).mean(dim=(-2,-1,0))).sum().cpu().numpy()
            umax = max(abs(run.u).max(), abs(run.v).max(), abs(run.w).max()).cpu().numpy()
            log_str = f't={t_:.3f}(s), dt={dt:.3f}(s), ke={ke:.3f}(m²/s²), umax={umax:.3f}(m/s).'
            print(log_str)
            next_log += step_log

        if step_plot > 0 and t >= next_plot:
            u = (run.u[0,1]).cpu().numpy()
            v = (run.v[0,1]).cpu().numpy()
            w = (run.w[0,1]).cpu().numpy()
            a[0].imshow(u, vmin=-um, vmax=um, **im_kwargs)
            a[1].imshow(v, vmin=-vm, vmax=vm, **im_kwargs)
            a[2].imshow(w, vmin=-wm, vmax=wm, **im_kwargs)
            plt.pause(0.5)
            next_plot += step_plot
        
        if step_save > 0 and t >= next_save:
            filename = os.path.join(outdir, f't_{count_save}.npz')
            np.savez(filename, t=t.cpu().numpy().astype('float32'),
                    u=run.u.cpu().numpy().astype('float32'),
                    v=run.v.cpu().numpy().astype('float32'),
                    w=run.w.cpu().numpy().astype('float32'))
            next_save += step_save
            count_save += 1
        
        t += run.dt
        run.compute_timestep()
