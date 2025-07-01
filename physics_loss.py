# -*- coding: utf-8 -*-
"""
Created on Tue 01 07 2025 01:53:56

@author: User
"""
# A function that computes the physics-based errors of a model pipeline dataset. 
#@tf.function()  #--Taking a  derivative from outside exposes the TensorArray to the boundary, and the conversion is not implemented in Tensorflow.
def physics_error_gas_2D(model,x,y):
    # 1D model adapted to 2D for fast computiation on graph mode
    with tf.device('/GPU:0'):                   # GPU is better if available
        dt_type=model.dtype
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        paddings = tf.constant([[0,0], [1, 1,], [1, 1],[0, 0]])
        paddings_dPVT=tf.constant([[0,0], [0,0], [1, 1,], [1, 1],[0, 0]])

        phi=tf.pad(nonormalize(model,x[4],stat_idx=4,compute=compute_),paddings,mode='SYMMETRIC')     
        # Add noise to the permeability
        # x[5]=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.1*normalize_diff(model,x[5],stat_idx=5,compute=True), dtype=dt_type)

        kx=tf.pad(nonormalize(model,x[5],stat_idx=5,compute=compute_),paddings,mode='SYMMETRIC') 
        ky=kx
        # kz=tf.pad(nonormalize(model,x[6],stat_idx=6,compute=compute_),paddings,mode='SYMMETRIC')     
         
        dx=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][0]
        dy=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][1]
        dz=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][2]

        dx_ij=dx[...,1:-1,1:-1,:]; dx_i1=dx[...,1:-1,2:,:]; dx_i_1=dx[...,1:-1,:-2,:]
        dy_ij=dy[...,1:-1,1:-1,:]; dy_j1=dy[...,2:,1:-1,:]; dy_j_1=dy[...,:-2,1:-1,:] 
        dz_ij=dz[...,1:-1,1:-1,:] 
        dv=(dx_ij*dy_ij*dz_ij)
        dx_avg_ih=(dx_i1+dx_ij)/2.; dx_avg_i_h=(dx_ij+dx_i_1)/2.   
        dy_avg_jh=(dy_j1+dy_ij)/2.; dy_avg_j_h=(dy_ij+dy_j_1)/2. 

        C=tf.constant(0.001127, dtype=model.dtype, shape=(), name='const1')
        D=tf.constant(5.6145833334, dtype=model.dtype, shape=(), name='const2')
        eps=tf.constant(1e-7, dtype=model.dtype, shape=(), name='epsilon')    
        
        # Create the Connection Index Tensor for the wells
        # Cast a tensor of 1 for the well block and 0 for other cells &(q_n1_ij==model.cfd_type['Init_Grate']). The tensorflow scatter and update function can be used
        q_well_idx=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], tf.ones_like(model.cfd_type['Init_Grate']), model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        q_t0_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Init_Grate'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        min_bhp_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Min_BHP'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        
        no_wells=tf.cast(tf.shape(model.cfd_type['Init_Grate']),model.dtype)
        area_ij=dx_ij*dy_ij
        area_res=tf.cast(tf.math.reduce_prod(model.cfd_type['Dimension']['Measurement'][:2]),model.dtype)
        hc=tf.constant(model.cfd_type['Completion_Ratio'], dtype=dt_type, shape=(), name='completion_ratio')                              # Completion ratio
        
        # Static parameters
        phi_n1_ij=phi[...,1:-1,1:-1,:]  
        kx_ij=kx[...,1:-1,1:-1,:]; kx_i1=kx[...,1:-1,2:,:]; kx_i_1=kx[...,1:-1,:-2,:]
        ky_ij=ky[...,1:-1,1:-1,:]; ky_j1=ky[...,2:,1:-1,:]; ky_j_1=ky[...,:-2,1:-1,:]
        kx_avg_ih=(2.*kx_i1*kx_ij)/(kx_i1+kx_ij); kx_avg_i_h=(2.*kx_ij*kx_i_1)/(kx_ij+kx_i_1)
        ky_avg_jh=(2.*ky_j1*ky_ij)/(ky_j1+ky_ij); ky_avg_j_h=(2.*ky_ij*ky_j_1)/(ky_ij+ky_j_1)
        ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx_ij,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy_ij,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
        rw=tf.constant(0.1905/2, dtype=dt_type, shape=(), name='rw')
        model.phi_0_ij=phi_n1_ij
        model.cf_ij=97.32e-6/(1+55.8721*model.phi_0_ij**1.428586)
        Sgi=tf.constant((1-model.cfd_type['SCAL']['End_Points']['Swmin']),dtype=model.dtype,shape=(),name='Sgi')
        Soi=1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sgi
        tmax=model.cfd_type['Max_Train_Time']
        Pi=model.cfd_type['Pi']
        invBgi=model.cfd_type['Init_InvBg'];dinvBgi=model.cfd_type['Init_DinvBg'];
        invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
        def shut_days(limits=None,time=None,dtype=None):
            return (tf.ones_like(time)-tf.cast((time>=limits[0])&(time<=limits[1]),dtype)) 
        
        # def wel_open(limits=None,time=None,dtype=None,eps=1.e-7):
        #     return tf.cast((time>=limits[1]-eps)&(time<=limits[1]+eps),dtype)
        
        Ck_ij=(2*(22/7)*hc*kx_ij*dz_ij*C)/(tf.math.log(ro/rw))
                        
        def physics_error_gas(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            out_stack_list=[0,1,2,3]
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            # shutins_idx=tf.reduce_mean([tf.reduce_mean([shut_days(limits=model.cfd_type['Connection_Shutins']['Days'][c][cidx],time=tn0,dtype=model.dtype) for cidx in model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)
            shutins_idx=dnn.conn_shutins_idx(tn0,model.cfd_type['Conn_Idx'],model.cfd_type['Connection_Shutins']['Days'])
            # welopens_idx=tf.reduce_mean([tf.reduce_mean([wel_open(limits=model.cfd_type['Connection_Shutins']['Days'][c][cidx],time=tn0,dtype=model.dtype) for cidx in model.cfd_type['Connection_Shutins']['Shutins_Per_Conn_Idx'][c]],axis=0) for c in model.cfd_type['Connection_Shutins']['Shutins_Idx']],axis=0)

            out_n0=model(xn0, training=True)
            out_n0,dPVT_n0,fac_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in out_stack_list]),[tf.pad(out_n0[i],paddings_dPVT,mode='SYMMETRIC') for i in [4,]],out_n0[-4:]

            # out_n0: predicted pressure output.
            # dPVT_n0: fluid property derivatives with respect to out_n0. 
            # predicted timestep at time point n0. 
            
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]
            
            invBg_n0_ij=out_n0[2][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBgug_n0_ij=(out_n0[2]*out_n0[3])[...,1:-1,1:-1,:]
            
            # Compute the average predicted timestep at time point n0.  
            tstep=(tf.reduce_mean(fac_n0[0],axis=[1,2,3],keepdims=True))
            
            # Normalize this timestep, as a difference, which is added to the nth time point to create the (n+1) prediction time point. 
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)

            # Create the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)

            # Re-evaluate the model time point n1. 
            out_n1,dPVT_n1,fac_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in out_stack_list]),[tf.pad(out_n1[i],paddings_dPVT,mode='SYMMETRIC') for i in [4,]],out_n1[-4:]
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]; 
            invBg_n1_ij=out_n1[2][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[3][...,1:-1,1:-1,:]
                       
            tstep_n1=tstep
            
            # Compute the average predicted timestep at time point n1.  
            tstep_n2=(tf.reduce_mean(fac_n1[0],axis=[1,2,3],keepdims=True))

            # Re-evaluate the model time point n2. 
            # However, the pressure and fluid properties at n2 are obtained by extrapolation
            p_n2_ij=(p_n1_ij-p_n0_ij)*(1.+tf.math.divide_no_nan(tstep_n2,tstep_n1))+p_n0_ij

            #=============================Relative Permeability Function=========================================================
            krog_n0,krgo_n0=krog_n1,krgo_n1=model.cfd_type['Kr_gas_oil'](Sgi)              #Entries: oil, and gas
            #====================================================================================================================
            #Define pressure variables 
            p_n1_i1=out_n1[0][...,1:-1,2:,:]; p_n1_i_1=out_n1[0][...,1:-1,:-2,:]
            p_n1_j1=out_n1[0][...,2:,1:-1,:]; p_n1_j_1=out_n1[0][...,:-2,1:-1,:]
            #====================================================================================================================
            # Compute d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision.
            d_dp_invBg_n0=dPVT_n0[0][0]
            invBgug_n1=(out_n1[2]*out_n1[3])
            invBgug_n1_ij=invBgug_n1[...,1:-1,1:-1,:]; 
            invBgug_n1_i1=invBgug_n1[...,1:-1,2:,:]; invBgug_n1_i_1=invBgug_n1[...,1:-1,:-2,:]
            invBgug_n1_j1=invBgug_n1[...,2:,1:-1,:]; invBgug_n1_j_1=invBgug_n1[...,:-2,1:-1,:]
            #====================================================================================================================
            # Compute the grid block pressures and fluid properties at faces using the average value function weighting
            p_n1_ih=(p_n1_i1+p_n1_ij)*0.5; p_n1_i_h=(p_n1_ij+p_n1_i_1)*0.5 
            p_n1_jh=(p_n1_j1+p_n1_ij)*0.5; p_n1_j_h=(p_n1_ij+p_n1_j_1)*0.5 
            p_n1_h=[p_n1_ih,p_n1_jh,p_n1_i_h,p_n1_j_h]           

            invBgug_avg_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_avg_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_avg_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_avg_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            cr_n0_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  #tf.zeros_like(phi)  
            cp_n1_ij=Sgi*((phi_n1_ij*d_dp_invBg_n0[...,1:-1,1:-1,:])+cr_n0_ij)

            a1_n1=C*kx_avg_i_h*krgo_n1*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(1/dx_ij)
            a2_n1=C*ky_avg_j_h*krgo_n1*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(1/dy_ij)
            a3_n1=C*kx_avg_ih*krgo_n1*invBgug_avg_n1_ih*(1/dx_avg_ih)*(1/dx_ij)
            a4_n1=C*ky_avg_jh*krgo_n1*invBgug_avg_n1_jh*(1/dy_avg_jh)*(1/dy_ij)
            a5_n1=(1/D)*(cp_n1_ij/(tstep))
            
            b1_n1=C*kx_avg_i_h*krgo_n1*invBgug_avg_n1_i_h*(1/dx_avg_i_h)*(dz_ij*dy_ij)
            b2_n1=C*ky_avg_j_h*krgo_n1*invBgug_avg_n1_j_h*(1/dy_avg_j_h)*(dz_ij*dx_ij)
            b3_n1=C*kx_avg_ih*krgo_n1*invBgug_avg_n1_ih*(1/dx_avg_ih)*(dz_ij*dy_ij)
            b4_n1=C*ky_avg_jh*krgo_n1*invBgug_avg_n1_jh*(1/dy_avg_jh)*(dz_ij*dx_ij)
            
            # Define grid weights
            well_wt=(tf.cast((q_well_idx==1),model.dtype)*1.)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            tsf_wt=tf.cast(xn0[3]<tsn['Time'],model.dtype)*1.+tf.cast(xn0[3]>=tsn['Time'],model.dtype)*tsn['Shift_Fac']
            # ===================================================================================================================
            # Compute bottom hole pressure
            q_n1_ij,_=fac_n1[-2],fac_n1[-1]
            
            # Compute the truncation Error term.
            trn_err=(dv/D)*cp_n1_ij*((2e-0*tf.keras.backend.epsilon()/tstep_n1)+(((tstep_n2*(p_n0_ij))+(tstep_n1*(p_n2_ij))-((tstep_n1+tstep_n2)*(p_n1_ij)))/((tstep_n1*tstep_n2)+tstep_n2**2.)))
            # ===================================================================================================================
            # Compute the domain loss.
            dom_divq_gg=dv*((-a1_n1*p_n1_i_1)+(-a2_n1*p_n1_j_1)+((a1_n1+a2_n1+a3_n1+a4_n1)*p_n1_ij)+(-a3_n1*p_n1_i1)+(-a4_n1*p_n1_j1)+(q_n1_ij/dv))
            dom_acc_gg=dv*a5_n1*(p_n1_ij-p_n0_ij)+trn_err
            dom=well_wt*(dom_divq_gg+dom_acc_gg)
            # Debugging....
            # tf.print('d_dp_invBg_n1\n',d_dp_invBg_n1,'InvBg_n0\n',invBg_n0,'InvBg_n1\n',invBg_n1,'InvUg_n1\n',invug_n1,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
            # tf.print('TIMESTEP\n',tf.reduce_mean(tstep,axis=[1,2,3]),output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/tstep.out" )
            #======================================= DBC Solution ===============================================================
            # Compute the external dirichlet boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            dbc=tf.zeros_like(dom)                 # Set at zero for now!
            #======================================= NBC Solution =============================================================== 
            # Compute the external Neumann boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            nbc=tf.zeros_like(dom)                 # Set at zero for now!
            #======================================= IBC Solution ===============================================================
            # Compute the inner boundary condition loss (wells).
            # shutins_wt=4*(1.-shutins_idx)+1.
            ibc_n=q_well_idx*((dom_divq_gg))       
            #======================================= Material Balance Check =====================================================
            # Compute the material balance loss. 
            kdims=False
            mbc=(-tf.reduce_sum(q_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(dv*Sgi*phi_n1_ij*(invBg_n1_ij-invBg_n0_ij)*(1/(D*tstep)),axis=[1,2,3],keepdims=kdims))
            #======================================= Cumulative Material Balance Check ==========================================
            # Optional array: Compute the cumulative material balance loss (this loss is not considered - set as zero)
            cmbc=tf.zeros_like(dom) 
            #======================================= Initial Condition ==========================================================
            # Optional array: Compute the initial condition loss. This loss is set as zero since it is already hard-enforced in the neural network layers. 
            ic=tf.zeros_like(dom)

            #=======================================================================================================
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom) #q_well_idx*(q_n1_ij-out_t1[-2])          # Rate output index: -2
            qrc_3=tf.zeros_like(dom)#q_well_idx*(1e-8*-out_t0[-2])
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            #=============================================================================
            return [dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]]
        
        # A function to stack the physics-based losses.
        def stack_physics_error():
            # Perform time point shifting (if necessary). Not used.
            x_i,tshift_fac_i,tsf_0_norm_i=time_shifting(model,x,shift_frac_mean=0.05,pred_cycle_mean=0.,random=False)
            tstep_wt=tf.cast(x_i[3]<=tsf_0_norm_i,model.dtype)+tf.cast(x_i[3]>tsf_0_norm_i,model.dtype)*tshift_fac_i

            # Dry Gas
            out_gg=physics_error_gas(model,x,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})
            dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=out_gg[0],out_gg[1],out_gg[2],out_gg[3],out_gg[4],out_gg[5],out_gg[6],out_gg[7],out_gg[8],out_gg[9]
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  
            return [dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i],[out_n0_i,out_n1_i],no_grid_blocks
        
        phy_error,out_n,no_blks=stack_physics_error()
        stacked_pinn_errors=phy_error[0:-2]
        stacked_outs=out_n
        checks=[phy_error[-2],phy_error[-1]]

        return stacked_pinn_errors,stacked_outs,checks,no_blks
    

def physics_error_gas_oil_2D(model,x,y):
    # 1D model adapted to 2D for fast computiation on graph mode
    with tf.device('/GPU:0'):                   # GPU is better if available
        dt_type=model.dtype
        #======================================================================================================
        # Compute the normalized values derivative--i.e., d(x_norm)/d(x)
        # Train statistics tensor: INDEX: {'x_coord', 'y_coord', 'z_coord', 'time', 'poro', 'permx', 'permz', 'grate',...}
        #                           KEYS: {'min', 'max', 'mean', 'std', 'count'}
        compute_=True
        
        paddings = tf.constant([[0,0], [1, 1,], [1, 1],[0, 0]])
        paddings_dPVT=tf.constant([[0,0], [0,0], [1, 1,], [1, 1],[0, 0]])
        phi=tf.pad(nonormalize(model,x[4],stat_idx=4,compute=compute_),paddings,mode='SYMMETRIC')     
        
        # Add noise to the permeability
        #x5=x[5]+tf.random.normal(shape=tf.shape(x[5]), mean=0.0, stddev=0.05, dtype=dt_type)

        kx=tf.pad(nonormalize(model,x[5],stat_idx=5,compute=compute_),paddings,mode='SYMMETRIC') 
        ky=kx
               
        dx=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][0]
        dy=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][1]
        dz=tf.ones_like(kx)*model.cfd_type['Dimension']['Gridblock_Dim'][2]

        dx_ij=dx[...,1:-1,1:-1,:]; dx_i1=dx[...,1:-1,2:,:]; dx_i_1=dx[...,1:-1,:-2,:]
        dy_ij=dy[...,1:-1,1:-1,:]; dy_j1=dy[...,2:,1:-1,:]; dy_j_1=dy[...,:-2,1:-1,:] 
        dz_ij=dz[...,1:-1,1:-1,:] 
        dv=(dx_ij*dy_ij*dz_ij)
        dx_avg_ih=(dx_i1+dx_ij)/2.; dx_avg_i_h=(dx_ij+dx_i_1)/2.   
        dy_avg_jh=(dy_j1+dy_ij)/2.; dy_avg_j_h=(dy_ij+dy_j_1)/2. 

        C=tf.constant(0.001127, dtype=model.dtype, shape=(), name='const1')
        D=tf.constant(5.6145833334, dtype=model.dtype, shape=(), name='const2')
        eps=tf.constant(1e-7, dtype=model.dtype, shape=(), name='epsilon')    
        
        # Create the Connection Index Tensor for the wells
        # Cast a tensor of 1 for the well block and 0 for other cells &(q_n1_ij==model.cfd_type['Init_Grate']). The tensorflow scatter and update function can be used
        q_well_idx=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], tf.ones_like(model.cfd_type['Init_Grate']), model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        q_t0_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Init_Grate'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])
        min_bhp_ij=tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], model.cfd_type['Min_BHP'], model.cfd_type['Dimension']['Dim']),0)*tf.ones_like(x[0])

        no_wells=tf.cast(tf.shape(model.cfd_type['Init_Grate']),model.dtype)
        area_ij=dx_ij*dy_ij
        area_res=tf.cast(tf.math.reduce_prod(model.cfd_type['Dimension']['Measurement'][:2]),model.dtype)
        hc=tf.constant(model.cfd_type['Completion_Ratio'], dtype=dt_type, shape=(), name='completion_ratio')                              # Completion ratio

        # Static parameters
        phi_n1_ij=phi[...,1:-1,1:-1,:]  
        kx_ij=kx[...,1:-1,1:-1,:]; kx_i1=kx[...,1:-1,2:,:]; kx_i_1=kx[...,1:-1,:-2,:]
        ky_ij=ky[...,1:-1,1:-1,:]; ky_j1=ky[...,2:,1:-1,:]; ky_j_1=ky[...,:-2,1:-1,:]
        kx_avg_ih=(2.*kx_i1*kx_ij)/(kx_i1+kx_ij); kx_avg_i_h=(2.*kx_ij*kx_i_1)/(kx_ij+kx_i_1)
        ky_avg_jh=(2.*ky_j1*ky_ij)/(ky_j1+ky_ij); ky_avg_j_h=(2.*ky_ij*ky_j_1)/(ky_ij+ky_j_1)
        ro=0.28*(tf.math.pow((((tf.math.pow(ky_ij/kx_ij,0.5))*(tf.math.pow(dx_ij,2)))+((tf.math.pow(kx_ij/ky_ij,0.5))*(tf.math.pow(dy_ij,2)))),0.5))/(tf.math.pow((ky_ij/kx_ij),0.25)+tf.math.pow((kx_ij/ky_ij),0.25))
        rw=tf.constant(0.1905/2, dtype=dt_type, shape=(), name='rw')
        model.phi_0_ij=phi_n1_ij
        model.cf_ij=97.32e-6/(1+55.8721*model.phi_0_ij**1.428586)
        Sgi=tf.constant((1-model.cfd_type['SCAL']['End_Points']['Swmin']),dtype=model.dtype,shape=(),name='Sgi')
        Soi=1-model.cfd_type['SCAL']['End_Points']['Swmin']-Sgi
        Sor=model.cfd_type['SCAL']['End_Points']['Sorg']
        tmax=model.cfd_type['Max_Train_Time']
        Pi=model.cfd_type['Pi']
        Pdew=model.cfd_type['Dew_Point']
        invBgi=model.cfd_type['Init_InvBg'];dinvBgi=model.cfd_type['Init_DinvBg'];
        invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
        rhg_std=model.cfd_type['Rhg_Std']
        rho_std=model.cfd_type['Rho_Std']
        
        # Define an optimal rate and BHP function
        Ck_ij=(2*(22/7)*hc*kx_ij*dz_ij*C)/(tf.math.log(ro/rw))

        # Normalized constants
        t0_norm=normalize(model,0.,stat_idx=3,compute=True)
        t1_norm=normalize(model,1.,stat_idx=3,compute=True)
        tmax_norm=normalize(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
        tmax_norm_diff=normalize_diff(model,model.cfd_type['Max_Train_Time'],stat_idx=3,compute=True)
        
        # Maximum Liquid Check
        # VroCVD_maxl=0.3
        # p_maxl=tf.constant(2600.,dtype=model.dtype,shape=(1,))   
        # PVT_maxl=model.PVT(tf.reshape(p_maxl,(-1,)))[0]
        # invBg_maxl=tf.reshape(PVT_maxl[0],(1,))
        # invBo_maxl=tf.reshape(PVT_maxl[1],(1,))      
        # Rs_maxl=tf.reshape(PVT_maxl[4],(1,)) 
        # Rv_maxl=tf.reshape(PVT_maxl[5],(1,))
    
        # boil_maxl=(VroCVD_maxl*invBo_maxl)+(1.-VroCVD_maxl)*(Rv_maxl*invBg_maxl)
        # bgas_maxl=(VroCVD_maxl*Rs_maxl*invBo_maxl)+(1.-VroCVD_maxl)*(invBg_maxl)
        # bo_bg_maxl=tf.math.divide_no_nan(boil_maxl,bgas_maxl)
                        
        def physics_error_gas_oil(model,xi,tsn={'Time':None,'Shift_Fac':1}):
            # nth step
            xn0=list(xi)
            tn0=nonormalize(model,xn0[3],stat_idx=3,compute=compute_) 
            out_n0=model(xn0, training=True)
            out_n0,dPVT_n0,fac_n0=tf.stack([tf.pad(out_n0[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8,9]]),[tf.pad(out_n0[i],paddings_dPVT,mode='SYMMETRIC') for i in [10]],out_n0[-4:]

            # out_n0: predicted pressure output.
            # dPVT_n0: fluid property derivatives with respect to out_n0. 
            # predicted timestep at time point n0. 
            
            p_n0_ij=out_n0[0][...,1:-1,1:-1,:]           
            Sg_n0_ij=out_n0[1][...,1:-1,1:-1,:]
            So_n0_ij=out_n0[2][...,1:-1,1:-1,:]
            invBg_n0_ij=out_n0[3][...,1:-1,1:-1,:]
            invBo_n0_ij=out_n0[4][...,1:-1,1:-1,:]
            invug_n0_ij=out_n0[5][...,1:-1,1:-1,:]
            invuo_n0_ij=out_n0[6][...,1:-1,1:-1,:]
            Rs_n0_ij=out_n0[7][...,1:-1,1:-1,:]
            Rv_n0_ij=out_n0[8][...,1:-1,1:-1,:]
            Vro_n0_ij=out_n0[9][...,1:-1,1:-1,:]
            invBgug_n0_ij=(out_n0[3]*out_n0[5])[...,1:-1,1:-1,:]
            invBouo_n0_ij=(out_n0[4]*out_n0[6])[...,1:-1,1:-1,:]
            RsinvBo_n0_ij=(out_n0[7]*out_n0[4])[...,1:-1,1:-1,:]
            RvinvBg_n0_ij=(out_n0[8]*out_n0[3])[...,1:-1,1:-1,:]
            
            invBgi,invBoi,invugi,invuoi,Rsi,Rvi=model.cfd_type['Init_InvBg'],model.cfd_type['Init_InvBo'],\
                model.cfd_type['Init_Invug'],model.cfd_type['Init_Invuo'],model.cfd_type['Init_Rs'],model.cfd_type['Init_Rv']
            invBgiugi=model.cfd_type['Init_InvBg']*model.cfd_type['Init_Invug']
            invBoiuoi=model.cfd_type['Init_InvBo']*model.cfd_type['Init_Invuo']
            RsinvBoi=model.cfd_type['Init_Rs']*model.cfd_type['Init_InvBo']
            RvinvBgi=model.cfd_type['Init_Rv']*model.cfd_type['Init_InvBg']
            
            krog_n0,krgo_n0=model.cfd_type['Kr_gas_oil'](out_n0[1])              #Entries: oil, and gas |out_n1[1]; sat_n1[0]
            krgo_n0_ij=krgo_n0[...,1:-1,1:-1,:]
            krog_n0_ij=krog_n0[...,1:-1,1:-1,:]

            # Compute the average predicted timestep at time point n0.  
            tstep=(tf.reduce_mean(fac_n0[0],axis=[1,2,3],keepdims=True))
            #tstep=(fac_n0[0][0])

            
            # Normalize this timestep, as a difference, which is added to the nth time point to create the (n+1) prediction time point. 
            tstep_norm=normalize_diff(model,tstep,stat_idx=3,compute=True)

            # Create the timestep (n+1)
            xn1=list(xi)
            xn1[3]+=tstep_norm
            tn1=nonormalize(model,xn1[3],stat_idx=3,compute=compute_) 
            out_n1=model(xn1, training=True)

            # Re-evaluate the model time point n1. 
            out_n1,fac_n1=tf.stack([tf.pad(out_n1[i],paddings,mode='SYMMETRIC') for i in [0,1,2,3,4,5,6,7,8,9]]),out_n1[-4:]           
            p_n1_ij=out_n1[0][...,1:-1,1:-1,:]
            Sg_n1_ij=out_n1[1][...,1:-1,1:-1,:]
            So_n1_ij=out_n1[2][...,1:-1,1:-1,:]
            invBg_n1_ij=out_n1[3][...,1:-1,1:-1,:]
            invBo_n1_ij=out_n1[4][...,1:-1,1:-1,:]
            invug_n1_ij=out_n1[5][...,1:-1,1:-1,:]
            invuo_n1_ij=out_n1[6][...,1:-1,1:-1,:]
            Rs_n1_ij=out_n1[7][...,1:-1,1:-1,:]
            Rv_n1_ij=out_n1[8][...,1:-1,1:-1,:]
            Vro_n1_ij=out_n1[9][...,1:-1,1:-1,:]
            invBgug_n1=(out_n1[3]*out_n1[5])
            invBouo_n1=(out_n1[4]*out_n1[6])
            RsinvBo_n1=(out_n1[7]*out_n1[4])
            RvinvBg_n1=(out_n1[8]*out_n1[3])
            RsinvBouo_n1=(out_n1[7]*out_n1[4]*out_n1[6])
            RvinvBgug_n1=(out_n1[8]*out_n1[3]*out_n1[5])
            invBgug_n1_ij=invBgug_n1[...,1:-1,1:-1,:]
            invBouo_n1_ij=invBouo_n1[...,1:-1,1:-1,:]
            RsinvBo_n1_ij=RsinvBo_n1[...,1:-1,1:-1,:]
            RvinvBg_n1_ij=RvinvBg_n1[...,1:-1,1:-1,:]   
            
            tstep_n1=tstep
            
            # Compute the average predicted timestep at time point n1.  
            dt1_wgt=1.#tf.reduce_mean(fac_n0[1],axis=[1,2,3],keepdims=True)
            tstep_n2=dt1_wgt*(tf.reduce_mean(fac_n1[0],axis=[1,2,3],keepdims=True))         # dt1_wgt*tstep_n1
            #tstep_n2=dt1_wgt*(fac_n1[0][0]) 
            dt1=tstep_n1; dt2=tstep_n2
            
            # Define a binary grid index indicating grid blocks above (0) or below (1) dew point/saturation pressure.
            tdew_idx=1#tf.cast((out_n1[0][...,1:-1,1:-1,:]<=model.cfd_type['Dew_Point']),model.dtype) #&(pg_n1_ij>=0.)

            # Re-evaluate the model time point n2. 
            # However, for a two-phase system, the mass accumuated (instead of pressure) and fluid properties at n2 are obtained by extrapolation
            # The mass accumulated per grid block is given as: 
            wt_mt=0.5                                # applying equal weights between the two phases.
            DM=(5.615/1000.)
            # mg_n0_ij=phi_n1_ij*((rhg_std*invBg_n0_ij*(1000/D)*Sg_n0_ij)+(rho_std*RvinvBg_n0_ij*Sg_n0_ij))    # reservoir mass flow: Conversion from bbl/Mscf to cf/scf=(5.615 ft/1000Scf)
            # mo_n0_ij=phi_n1_ij*((rho_std*invBo_n0_ij*So_n0_ij)+(rhg_std*RsinvBo_n0_ij*(1000/D)*So_n0_ij))
            # mt_n0_ij=mg_n0_ij+mo_n0_ij
            
            # mg_n1_ij=phi_n1_ij*((rhg_std*invBg_n1_ij*(1000/D)*Sg_n1_ij)+(rho_std*RvinvBg_n1_ij*Sg_n1_ij))
            # mo_n1_ij=phi_n1_ij*((rho_std*invBo_n1_ij*So_n1_ij)+(rhg_std*RsinvBo_n1_ij*(1000/D)*So_n1_ij))

            # mt_n1_ij=mg_n1_ij+mo_n1_ij
            # mt_n2_ij=(mt_n1_ij-mt_n0_ij)*(1.+tf.math.divide_no_nan(dt2,dt1))+mt_n0_ij
            
            mg_n0_ij=phi_n1_ij*((invBg_n0_ij*Sg_n0_ij)+(tdew_idx*RsinvBo_n0_ij*So_n0_ij))    # Surface Gas Conversion from bbl/Mscf to cf/scf=(5.615 ft/1000Scf)
            mo_n0_ij=phi_n1_ij*((tdew_idx*invBo_n0_ij*So_n0_ij)+(RvinvBg_n0_ij*Sg_n0_ij))    # Surface Oil Conversion
            mt_n0_ij=mg_n0_ij+mo_n0_ij
            mg_n1_ij=phi_n1_ij*((invBg_n1_ij*Sg_n1_ij)+(tdew_idx*RsinvBo_n1_ij*So_n1_ij))    # Surface Gas Conversion from bbl/Mscf to cf/scf=(5.615 ft/1000Scf)
            mo_n1_ij=phi_n1_ij*((tdew_idx*invBo_n1_ij*So_n1_ij)+(RvinvBg_n1_ij*Sg_n1_ij))    # Surface Oil Conversion
            mt_n1_ij=mg_n1_ij+mo_n1_ij
            mg_n2_ij=(mg_n1_ij-mg_n0_ij)*(1.+tf.math.divide_no_nan(dt2,dt1))+mg_n0_ij
            mo_n2_ij=(mo_n1_ij-mo_n0_ij)*(1.+tf.math.divide_no_nan(dt2,dt1))+mo_n0_ij
            mt_n2_ij=mg_n2_ij+mo_n2_ij
            
            
            # Compute the truncation error term.           
            # trn_err=(dv/D)*((2e-0*tf.keras.backend.epsilon()/dt1)+(((dt2*(mt_n0_ij))+(dt1*(mt_n2_ij))-((dt1+dt2)*(mt_n1_ij)))/((dt1*dt2)+dt2**2.)))
            # rte=(tf.reduce_mean(fac_n1[1],axis=[1,2,3],keepdims=True))-(tf.reduce_mean(fac_n0[1],axis=[1,2,3],keepdims=True))
            w1=10.
            rte=tf.keras.backend.epsilon()*(1/4)               # Average rounding error = (1/4) machine precison for a uniform distribution of float values
            trn_err_g=(dv/D)*((rte/dt1)+(((dt2*(mg_n0_ij))+(dt1*(mg_n2_ij))-((dt1+dt2)*(mg_n1_ij)))/((dt1*dt2)+dt2**2.)))
            trn_err_o=(dv/D)*((rte/dt1)+(((dt2*(mo_n0_ij))+(dt1*(mo_n2_ij))-((dt1+dt2)*(mo_n1_ij)))/((dt1*dt2)+dt2**2.)))
            
            # # Reparametrization Trick
            # e1=tf.math.log(dt1)
            # e2=tf.math.log(dt2)
            # mg0=tf.math.log(mg_n0_ij); mo0=tf.math.log(mo_n0_ij)
            # mg1=tf.math.log(mg_n1_ij); mo1=tf.math.log(mo_n1_ij)
            # mg2=tf.math.log(mg_n2_ij); mo2=tf.math.log(mo_n2_ij)
            
            # trn_err_g=(dv/D)*((rte/tf.math.exp(e1))+(((tf.math.exp(e2+mg0))+(tf.math.exp(e1+mg2))-((e1+e2)*tf.math.exp(mg1)))/(tf.math.exp(e1+e2)+tf.math.exp(2*e2))))
            # trn_err_o=(dv/D)*((rte/tf.math.exp(e1))+(((tf.math.exp(e2+mo0))+(tf.math.exp(e1+mo2))-((e1+e2)*tf.math.exp(mo1)))/(tf.math.exp(e1+e2)+tf.math.exp(2*e2))))

            # trn_err_g=(dv/D)*(((dt1*dt2)+dt2**2.)*(rte/dt1)+((dt2*(mg_n0_ij))+(dt1*(mg_n2_ij))-((dt1+dt2)*(mg_n1_ij))))
            # trn_err_o=(dv/D)*(((dt1*dt2)+dt2**2.)*(rte/dt1)+((dt2*(mo_n0_ij))+(dt1*(mo_n2_ij))-((dt1+dt2)*(mo_n1_ij))))

            # trn_err=trn_err_g+trn_err_o
            # ============================================ Relative Permeability ======================================
            krog_n1,krgo_n1=model.cfd_type['Kr_gas_oil'](out_n1[1])              #Entries: oil, and gas |out_n1[1]; sat_n1[0]
            krgo_n1_ij=krgo_n1[...,1:-1,1:-1,:]
            krog_n1_ij=krog_n1[...,1:-1,1:-1,:]
            # =========================================================================================================
            # Compute bottom hole pressure and rates
            # qfg_n1_ij,qdg_n1_ij,qfo_n1_ij,qvo_n1_ij,pwf_n1_ij=compute_rate_bhp_gas_oil(p_n0_ij,Sg_n0_ij,_invBg_n1_ij,_invBo_n1_ij,invBgug_n0_ij,invBouo_n0_ij,Rs_n0_ij,Rv_n0_ij,krgo_n0_ij,\
            #                                                     krog_n0_ij,q_t0_ij,min_bhp_ij,Ck_ij,q_well_idx,_Sgi=Sgi,_p_dew=Pdew,_shutins_idx=1,_ctrl_mode=1,\
            #                                                         _lmd=fac_n0[1],pre_sat_model=None,rel_perm_model=model.cfd_type['Kr_gas_oil'],model_PVT=model.PVT)
            qfg_n1_ij,qdg_n1_ij,qfo_n1_ij,qvo_n1_ij,pwf_n1_ij=fac_n1[-2][0],fac_n1[-2][1],fac_n1[-2][2],fac_n1[-2][3],fac_n1[-1]
            # qfg_n1_ij,qdg_n1_ij,qfo_n1_ij,qvo_n1_ij,pwf_n1_ij=fac_n0[-2][0],fac_n0[-2][1],fac_n0[-2][2],fac_n0[-2][3],fac_n0[-1]
            # =========================================================================================================
            # Compute the chord slopes for pressure and saturation. d_dp_Sg;d_dp_invBg at p(n+1) using the chord slope  -- Checks for nan (0./0.) when using low precision           
            _d_dpg_Sg_n1_ij=tf.math.divide_no_nan((out_n1[1]-out_n0[1]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            _d_dpo_So_n1_ij=tf.math.divide_no_nan((out_n1[2]-out_n0[2]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            d_dpg_Sg_n1_ij,d_dpo_So_n1_ij=_d_dpg_Sg_n1_ij,_d_dpo_So_n1_ij            
                        
            # Define gas and oil pressure variables for the grid blocks and those of adjoining grid blocks. 
            # In the absence of capillary pressures, the gas pressure is equal to the oil pressure, i.e., pg=po
            pg_n0_ij=out_n0[0][...,1:-1,1:-1,:]; po_n0_ij=out_n0[0][...,1:-1,1:-1,:];
            pg_n1_ij=out_n1[0][...,1:-1,1:-1,:]; po_n1_ij=out_n1[0][...,1:-1,1:-1,:];
            pg_n1_i1=out_n1[0][...,1:-1,2:,:]; po_n1_i1=out_n1[0][...,1:-1,2:,:];
            pg_n1_i_1=out_n1[0][...,1:-1,:-2,:]; po_n1_i_1=out_n1[0][...,1:-1,:-2,:];   
            pg_n1_j1=out_n1[0][...,2:,1:-1,:]; po_n1_j1=out_n1[0][...,2:,1:-1,:];
            pg_n1_j_1=out_n1[0][...,:-2,1:-1,:]; po_n1_j_1=out_n1[0][...,:-2,1:-1,:];
                        
            # Define gas and oil fluid property and derivative variables of the grid blocks and those of adjoining grid blocks.
            invBgug_n1_i1=invBgug_n1[...,1:-1,2:,:]; invBgug_n1_i_1=invBgug_n1[...,1:-1,:-2,:]
            invBgug_n1_j1=invBgug_n1[...,2:,1:-1,:]; invBgug_n1_j_1=invBgug_n1[...,:-2,1:-1,:]
            
            invBouo_n1_ij=invBouo_n1[...,1:-1,1:-1,:]; 
            invBouo_n1_i1=invBouo_n1[...,1:-1,2:,:]; invBouo_n1_i_1=invBouo_n1[...,1:-1,:-2,:]
            invBouo_n1_j1=invBouo_n1[...,2:,1:-1,:]; invBouo_n1_j_1=invBouo_n1[...,:-2,1:-1,:]
            
            RvinvBgug_n1_ij=RvinvBgug_n1[...,1:-1,1:-1,:]; 
            RvinvBgug_n1_i1=RvinvBgug_n1[...,1:-1,2:,:]; RvinvBgug_n1_i_1=RvinvBgug_n1[...,1:-1,:-2,:]
            RvinvBgug_n1_j1=RvinvBgug_n1[...,2:,1:-1,:]; RvinvBgug_n1_j_1=RvinvBgug_n1[...,:-2,1:-1,:]

            RsinvBouo_n1_ij=RsinvBouo_n1[...,1:-1,1:-1,:]; 
            RsinvBouo_n1_i1=RsinvBouo_n1[...,1:-1,2:,:]; RsinvBouo_n1_i_1=RsinvBouo_n1[...,1:-1,:-2,:]
            RsinvBouo_n1_j1=RsinvBouo_n1[...,2:,1:-1,:]; RsinvBouo_n1_j_1=RsinvBouo_n1[...,:-2,1:-1,:]
            
            #Derivatives. 
            # d_dpg_invBg_n1_ij=tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_invBg_n1_ij=tf.math.divide_no_nan((out_n1[3]-out_n0[3]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpg_invBo_n1_ij=tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_invBo_n1_ij=tf.math.divide_no_nan((out_n1[4]-out_n0[4]),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            # d_dpg_RsinvBo_n1_ij=tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_RsinvBo_n1_ij=tf.math.divide_no_nan(((out_n1[7]*out_n1[4])-(out_n0[7]*out_n0[4])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpg_RvinvBg_n1_ij=tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]
            # d_dpo_RvinvBg_n1_ij=tf.math.divide_no_nan(((out_n1[8]*out_n1[3])-(out_n0[8]*out_n0[3])),(out_n1[0]-out_n0[0]))[...,1:-1,1:-1,:]

            d_dpg_invBg_n1_ij=dPVT_n0[0][0][...,1:-1,1:-1,:]
            d_dpo_invBg_n1_ij=dPVT_n0[0][0][...,1:-1,1:-1,:]
            d_dpg_invBo_n1_ij=dPVT_n0[0][1][...,1:-1,1:-1,:]
            d_dpo_invBo_n1_ij=dPVT_n0[0][1][...,1:-1,1:-1,:]

            d_dpg_RsinvBo_n1_ij=((out_n0[7]*dPVT_n0[0][1])+(out_n0[4]*dPVT_n0[0][4]))[...,1:-1,1:-1,:] 
            d_dpo_RsinvBo_n1_ij=((out_n0[7]*dPVT_n0[0][1])+(out_n0[4]*dPVT_n0[0][4]))[...,1:-1,1:-1,:]
            d_dpg_RvinvBg_n1_ij=((out_n0[8]*dPVT_n0[0][0])+(out_n0[3]*dPVT_n0[0][5]))[...,1:-1,1:-1,:]
            d_dpo_RvinvBg_n1_ij=((out_n0[8]*dPVT_n0[0][0])+(out_n0[3]*dPVT_n0[0][5]))[...,1:-1,1:-1,:]
            
            # Compute the fluid property variables at the grid block faces using the average value function weighting.         
            invBgug_n1_ih=(invBgug_n1_i1+invBgug_n1_ij)/2.; invBgug_n1_i_h=(invBgug_n1_ij+invBgug_n1_i_1)/2.
            invBgug_n1_jh=(invBgug_n1_j1+invBgug_n1_ij)/2.; invBgug_n1_j_h=(invBgug_n1_ij+invBgug_n1_j_1)/2.
            invBouo_n1_ih=(invBouo_n1_i1+invBouo_n1_ij)/2.; invBouo_n1_i_h=(invBouo_n1_ij+invBouo_n1_i_1)/2.
            invBouo_n1_jh=(invBouo_n1_j1+invBouo_n1_ij)/2.; invBouo_n1_j_h=(invBouo_n1_ij+invBouo_n1_j_1)/2.
            
            RvinvBgug_n1_ih=(RvinvBgug_n1_i1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_i_h=(RvinvBgug_n1_ij+RvinvBgug_n1_i_1)/2.
            RvinvBgug_n1_jh=(RvinvBgug_n1_j1+RvinvBgug_n1_ij)/2.; RvinvBgug_n1_j_h=(RvinvBgug_n1_ij+RvinvBgug_n1_j_1)/2.
            RsinvBouo_n1_ih=(RsinvBouo_n1_i1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_i_h=(RsinvBouo_n1_ij+RsinvBouo_n1_i_1)/2.
            RsinvBouo_n1_jh=(RsinvBouo_n1_j1+RsinvBouo_n1_ij)/2.; RsinvBouo_n1_j_h=(RsinvBouo_n1_ij+RsinvBouo_n1_j_1)/2.
             
            # Compute the gas and oil relative permeability variables at the grid block faces. 
            # The upstream weighting is suitable for saturation-dependent terms like relative permeability. 
            # Only the upstream weighting method works for linearization of saturation dependent terms in numerical simulations; the
            # average function value weighting gives erroneus results (Abou-Kassem, 2006).
            krgo_n1_i1=krgo_n1[...,1:-1,2:,:]; krgo_n1_i_1=krgo_n1[...,1:-1,:-2,:]
            krgo_n1_j1=krgo_n1[...,2:,1:-1,:]; krgo_n1_j_1=krgo_n1[...,:-2,1:-1,:]    
            
            krog_n1_i1=krog_n1[...,1:-1,2:,:]; krog_n1_i_1=krog_n1[...,1:-1,:-2,:]
            krog_n1_j1=krog_n1[...,2:,1:-1,:]; krog_n1_j_1=krog_n1[...,:-2,1:-1,:]
            
            #For i to be upstream (i+1), pot_i<=0; i to be downstream (i+1), pot_i>0.
            potg_n1_i1=poto_n1_i1=(pg_n1_i1-pg_n1_ij)
            potg_n1_i_1=poto_n1_i_1=(pg_n1_ij-pg_n1_i_1)
            potg_n1_j1=poto_n1_j1=(pg_n1_j1-pg_n1_ij)
            potg_n1_j_1=poto_n1_j_1=(pg_n1_ij-pg_n1_j_1)
            
            krgo_n1_ih=tf.cast(potg_n1_i1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_i1>0.,model.dtype)*krgo_n1_i1
            krgo_n1_i_h=tf.cast(potg_n1_i_1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_i_1>0.,model.dtype)*krgo_n1_i_1
            krgo_n1_jh=tf.cast(potg_n1_j1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_j1>0.,model.dtype)*krgo_n1_j1
            krgo_n1_j_h=tf.cast(potg_n1_j_1<=0.,model.dtype)*krgo_n1_ij+tf.cast(potg_n1_j_1>0.,model.dtype)*krgo_n1_j_1

            krog_n1_ih=tf.cast(poto_n1_i1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_i1>0.,model.dtype)*krog_n1_i1
            krog_n1_i_h=tf.cast(poto_n1_i_1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_i_1>0.,model.dtype)*krog_n1_i_1
            krog_n1_jh=tf.cast(poto_n1_j1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_j1>0.,model.dtype)*krog_n1_j1
            krog_n1_j_h=tf.cast(poto_n1_j_1<=0.,model.dtype)*krog_n1_ij+tf.cast(poto_n1_j_1>0.,model.dtype)*krog_n1_j_1
            
            # krgo_n1_ih=krgo_n1_i_h=krgo_n1_jh=krgo_n1_j_h=krgo_n1_ij
            # krog_n1_ih=krog_n1_i_h=krog_n1_jh=krog_n1_j_h=krgo_n1_ij

            # Compute the rock compressibility term. This is the product of the rock compressibility, porosity and inverse formation volume factor at n0.
            cprgg_n0_ij=(model.phi_0_ij*model.cf*invBg_n0_ij)  
            cprgo_n0_ij=(model.phi_0_ij*model.cf*RsinvBo_n0_ij)  
            cproo_n0_ij=(model.phi_0_ij*model.cf*invBo_n0_ij)  
            cprog_n0_ij=(model.phi_0_ij*model.cf*RvinvBg_n0_ij)  
            
            # Compute variables for the gas phase flow -- free gas in the gas phase, and gas dissolved in the oil phase.
            agg_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*invBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            agg_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*invBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            ago_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*RsinvBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            ago_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*RsinvBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            agg_n1_ih=C*kx_avg_ih*(krgo_n1_ih*invBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            agg_n1_jh=C*ky_avg_jh*(krgo_n1_jh*invBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            ago_n1_ih=C*kx_avg_ih*(krog_n1_ih*RsinvBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            ago_n1_jh=C*ky_avg_jh*(krog_n1_jh*RsinvBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpgg_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_invBg_n1_ij)+cprgg_n0_ij))*(pg_n1_ij-pg_n0_ij)
            cpgo_n1_ij=(1/(D*tstep))*((phi_n1_ij*RsinvBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_RsinvBo_n1_ij)+cprgo_n0_ij))*(po_n1_ij-po_n0_ij)
            
            # Compute variables for the oil phase flow -- free oil in the oil phase, and oil vapourized in the gas phase.
            aoo_n1_i_h=C*kx_avg_i_h*(krog_n1_i_h*invBouo_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aoo_n1_j_h=C*ky_avg_j_h*(krog_n1_j_h*invBouo_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aog_n1_i_h=C*kx_avg_i_h*(krgo_n1_i_h*RvinvBgug_n1_i_h)*(1/dx_avg_i_h)*(1/dx_ij)
            aog_n1_j_h=C*ky_avg_j_h*(krgo_n1_j_h*RvinvBgug_n1_j_h)*(1/dy_avg_j_h)*(1/dy_ij)
            aoo_n1_ih=C*kx_avg_ih*(krog_n1_ih*invBouo_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aoo_n1_jh=C*ky_avg_jh*(krog_n1_jh*invBouo_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)
            aog_n1_ih=C*kx_avg_ih*(krgo_n1_ih*RvinvBgug_n1_ih)*(1/dx_avg_ih)*(1/dx_ij)
            aog_n1_jh=C*ky_avg_jh*(krgo_n1_jh*RvinvBgug_n1_jh)*(1/dy_avg_jh)*(1/dy_ij)

            cpoo_n1_ij=(1/(D*tstep))*((phi_n1_ij*invBo_n1_ij*d_dpo_So_n1_ij)+So_n0_ij*((phi_n1_ij*d_dpo_invBo_n1_ij)+cproo_n0_ij))*(po_n1_ij-po_n0_ij)
            cpog_n1_ij=(1/(D*tstep))*((phi_n1_ij*RvinvBg_n1_ij*d_dpg_Sg_n1_ij)+Sg_n0_ij*((phi_n1_ij*d_dpg_RvinvBg_n1_ij)+cprog_n0_ij))*(pg_n1_ij-pg_n0_ij)

            # Compute the domain loss. 
            # Domain divergence terms for the gas flow. 
            dom_divq_gg=dv*((-agg_n1_i_h*pg_n1_i_1)+(-agg_n1_j_h*pg_n1_j_1)+((agg_n1_i_h+agg_n1_j_h+agg_n1_ih+agg_n1_jh)*pg_n1_ij)+\
                      (-agg_n1_ih*pg_n1_i1)+(-agg_n1_jh*pg_n1_j1)+(qfg_n1_ij/dv))
                
            dom_divq_go=dv*((-ago_n1_i_h*po_n1_i_1)+(-ago_n1_j_h*po_n1_j_1)+((ago_n1_i_h+ago_n1_j_h+ago_n1_ih+ago_n1_jh)*po_n1_ij)+\
                      (-ago_n1_ih*po_n1_i1)+(-ago_n1_jh*po_n1_j1)+(qdg_n1_ij/dv))
                
            # Domain accumulation terms for the gas flow. 
            dom_acc_gg=dv*(cpgg_n1_ij)
            dom_acc_go=dv*(cpgo_n1_ij) 
            
            dom_gg=dom_divq_gg+dom_acc_gg
            dom_go=tdew_idx*(dom_divq_go+dom_acc_go)
            
            # Domain loss for the gas flow. 
            dom_g=dom_gg+dom_go

            # Domain divergence terms for the oil flow. 
            dom_divq_oo=dv*((-aoo_n1_i_h*po_n1_i_1)+(-aoo_n1_j_h*po_n1_j_1)+((aoo_n1_i_h+aoo_n1_j_h+aoo_n1_ih+aoo_n1_jh)*po_n1_ij)+\
                      (-aoo_n1_ih*po_n1_i1)+(-aoo_n1_jh*po_n1_j1)+(qfo_n1_ij/dv))
                
            dom_divq_og=dv*((-aog_n1_i_h*pg_n1_i_1)+(-aog_n1_j_h*pg_n1_j_1)+((aog_n1_i_h+aog_n1_j_h+aog_n1_ih+aog_n1_jh)*pg_n1_ij)+\
                      (-aog_n1_ih*pg_n1_i1)+(-aog_n1_jh*pg_n1_j1)+(qvo_n1_ij/dv))
                
            # Domain accumulation terms for the oil flow. 
            dom_acc_oo=dv*(cpoo_n1_ij) 
            dom_acc_og=dv*(cpog_n1_ij)
            
            dom_oo=tdew_idx*(dom_divq_oo+dom_acc_oo)
            dom_og=dom_divq_og+dom_acc_og
            
            # Domain loss for the oil flow. 
            dom_o=dom_oo+dom_og

            well_wt=(tf.cast((q_well_idx==1),model.dtype)*1.)+(tf.cast((q_well_idx!=1),model.dtype)*1.)
            # Debugging...
            #tf.print('d_dpg_invBg_n1_ij\n',d_dpg_invBg_n1_ij,'InvBg_n0\n',invBg_n0_ij,'InvBg_n1\n',invBg_n1_ij,'InvUg_n1\n',invug_n1_ij,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/pre_pvt.out" )
            
            # Weighting values for the gas and oil flows -- equal weights appied, i.e., 0.5
            wt_dom=0.5

            # Compute the Kullback-Leibler (KL) Divergence
            # kl_loss_0 = -0.5 * tf.reduce_mean(1 + fac_n0[0][2] - tf.square(fac_n0[0][1]) - tf.exp(fac_n0[0][2]), axis=[1,2,3])
            # kl_loss_1 = -0.5 * tf.reduce_mean(1 + fac_n1[0][2] - tf.square(fac_n1[0][1]) - tf.exp(fac_n1[0][2]), axis=[1,2,3])
            # kl_loss=tf.stack([kl_loss_0,kl_loss_1])
            # Total domain loss for the gas and oil flows. 
            #dom=wt_dom*(dom_gg+dom_go)+(1.-wt_dom)*(dom_oo+dom_og)+trn_err
            # Compute pressure change
            trn_err=(trn_err_g+trn_err_o)
            dom=(dom_gg+dom_go)+(dom_oo+dom_og)#+trn_err
            # ====================================== DBC Solution ===============================================================
            # Compute the external dirichlet boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            dbc=tf.zeros_like(dom)                  # Set at zero for now!
            # ====================================== NBC Solution =============================================================== 
            # Compute the external Neumann boundary loss (set as zero, since its already computed in the main grid by image grid blocks)
            nbc=tf.zeros_like(dom)                  # Set at zero for now!
            # ====================================== IBC Solution ===============================================================
            # Compute the inner boundary condition loss (wells).
            # regu=0.0005*tf.linalg.global_norm(model.trainable_variables)**2
            wt_ibc=0.5                              # Equal weights between the gas and oil flows. 
            #ibc_n=q_well_idx*(wt_ibc*(dom_divq_gg+dom_divq_go)+(1.-wt_ibc)*(dom_divq_oo+dom_divq_og))
            ibc_n=q_well_idx*((dom_divq_gg+tdew_idx*dom_divq_go)+(tdew_idx*dom_divq_oo+dom_divq_og))
            # ====================================== Material Balance Check =====================================================
            # Compute the material balance loss. 
            kdims=False
            wt_mbc=0.5                              # Equal weights between the gas and oil flows. 
            mbc_gg=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*invBg_n1_ij)-(Sg_n0_ij*invBg_n0_ij))
            mbc_go=tdew_idx*dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*RsinvBo_n1_ij)-(So_n0_ij*RsinvBo_n0_ij))
            
            mbc_oo=tdew_idx*dv*(1/(D*tstep))*phi_n1_ij*((So_n1_ij*invBo_n1_ij)-(So_n0_ij*invBo_n0_ij))
            mbc_og=dv*(1/(D*tstep))*phi_n1_ij*((Sg_n1_ij*RvinvBg_n1_ij)-(Sg_n0_ij*RvinvBg_n0_ij))
            
            mbc_g=(-tf.reduce_sum(qfg_n1_ij+tdew_idx*qdg_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_gg+mbc_go,axis=[1,2,3],keepdims=kdims))             
            mbc_o=(-tf.reduce_sum(tdew_idx*qfo_n1_ij+qvo_n1_ij,axis=[1,2,3],keepdims=kdims)-tf.reduce_sum(mbc_oo+mbc_og,axis=[1,2,3],keepdims=kdims))
            
            
            mbc=(mbc_g)+(mbc_o)
            # ====================================== Cumulative Material Balance Check ==========================================
            # Optional array: Compute the cumulative material balance loss (this loss is not considered - set as zero)
            # Check the Maximum Liquid saturation is always conserved
            # VrgCVD_n0_ij=tf.math.divide_no_nan(Sg_n0_ij,Sgi)
            # VroCVD_n0_ij=tf.math.divide_no_nan(So_n0_ij,Sgi)
            # boil_n0_ij=(VroCVD_n0_ij*invBo_n0_ij)+(VrgCVD_n0_ij)*(RvinvBg_n0_ij)
            # bgas_n0_ij=(VroCVD_n0_ij*RsinvBo_n1_ij)+(VrgCVD_n0_ij)*(invBg_n0_ij)
            # beq_n0_ij=dv*(1/(D*tstep))*phi_n1_ij*(boil_n0_ij-(bo_bg_maxl*bgas_n0_ij))

            # VrgCVD_n1_ij=tf.math.divide_no_nan(Sg_n1_ij,Sgi)
            # VroCVD_n1_ij=tf.math.divide_no_nan(So_n1_ij,Sgi)
            # boil_n1_ij=(VroCVD_n1_ij*invBo_n1_ij)+(VrgCVD_n1_ij)*(RvinvBg_n1_ij)
            # bgas_n1_ij=(VroCVD_n1_ij*RsinvBo_n1_ij)+(VrgCVD_n1_ij)*(invBg_n1_ij)
            # beq_n1_ij=dv*(1/(D*tstep))*phi_n1_ij*(boil_n1_ij-(bo_bg_maxl*bgas_n1_ij))
            cmbc=(trn_err)#tf.reduce_sum(trn_err,axis=[1,2,3],keepdims=kdims)#tf.zeros_like(dom) 
            
            # ======================================= Initial Condition =========================================================
            # Optional array: Compute the initial condition loss. This loss is set as zero since it is already hard-enforced in the neural network layers. 
            ic=tf.zeros_like(dom)
            # ====================================== Other Placeholders =========================================================
            # Supports up to four loss variables. Not currently used. 
            qrc_1=tf.zeros_like(dom)
            qrc_2=tf.zeros_like(dom)
            qrc_3=tf.zeros_like(dom)
            qrc_4=tf.zeros_like(dom)
            qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
            # ===================================================================================================================
            return [dom,dbc,nbc,ibc_n,ic,qrc,mbc,cmbc,out_n0[...,:,1:-1,1:-1,:],out_n1[...,:,1:-1,1:-1,:]]

        # Stack the physics-based loss (if any)
        def stack_physics_error():
            x_i,tshift_fac_i,tsf_0_norm_i=time_shifting(model,x,shift_frac_mean=0.05,pred_cycle_mean=0.,random=False)
            tstep_wt=tf.cast(x_i[3]<=tsf_0_norm_i,model.dtype)+tf.cast(x_i[3]>tsf_0_norm_i,model.dtype)*tshift_fac_i

            # Gas-Oil
            out_go=physics_error_gas_oil(model,x_i,tsn={'Time':tsf_0_norm_i,'Shift_Fac':1.})    #No shift of time
            dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i,out_n0_i,out_n1_i=out_go[0],out_go[1],out_go[2],out_go[3],out_go[4],out_go[5],out_go[6],out_go[7],out_go[8],out_go[9]
         
            no_grid_blocks=[0.,0.,tf.reduce_sum(q_well_idx),tf.reduce_sum(q_well_idx),0.]  #update later
            return [dom_i,dbc_i,nbc_i,ibc_n_i,ic_i,qrc_i,mbc_i,cmbc_i],[out_n0_i,out_n1_i],no_grid_blocks
        
        phy_error,out_n,no_blks=stack_physics_error()
        stacked_pinn_errors=phy_error[0:-2]
        stacked_outs=out_n 
        checks=[phy_error[-2],phy_error[-1]]

        return stacked_pinn_errors,stacked_outs,checks,no_blks
    
# A function that outputs zeros for the physics-based losses, preventing the demanding physics-based computations. Used during non-physics-based supervised learning. 
#@tf.function
def zeros_like_pinn_error(model,x,y):
    out_n0=model.loss_func['Squeeze_Out'](tf.stack(model(x, training=True)))
    dom=tf.zeros_like(y[0],dtype=dt_type)
    dbc=dom
    nbc=dom
    ibc_n=dom
    ic=dom
    mbc=dom
    cmbc=dom
    qrc_1=dom;qrc_2=dom;qrc_3=dom;qrc_4=dom
    out_n1=out_n0
    no_grid_blocks=[0.,0.,0.,0.,0.]
    qrc=[qrc_1,qrc_2,qrc_3,qrc_4]
    return [dom,dbc,nbc,ibc_n,ic,qrc,],[out_n0,out_n1],[mbc,cmbc],no_grid_blocks
        
# ===============================================================================================================================
@tf.function
def boolean_mask_cond(x=None,y=None,data=[],bool_mask=[],solu_count=None):
    output=tf.cond(tf.math.equal(x,y),lambda: [tf.boolean_mask(data,bool_mask,axis=0),tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0))],\
                           lambda: [tf.multiply(tf.boolean_mask(data,bool_mask,axis=0),0.),tf.multiply(tf.ones_like(tf.boolean_mask(data,bool_mask,axis=0)),0.)])
    return output

# ================================================= Training Loss Computation ===================================================
# A function that computes the training loss of either non-physics-based supervised learning or physics-based semi-supervised learning. 
# Tensorflow graph mode (@tf.function), also accelerated linear algebra XLA (jit_compile=True) is utilized to improve the speed of computation.
@tf.function(jit_compile=True)
def pinn_batch_sse_grad(model,x,y):
    # Physics gradient for Arrangement Type 1: 
    # DATA ARRANGEMENT FOR TYPE 1
    # Training Features: Model Inputs: 
    # x is a list of inputs (numpy arrays or tensors) for the given batch_size
    # x[0] is the grid block x_coord in ft.
    # x[1] is the grid block y_coord in ft.
    # x[2] is the grid block z_coord in ft.
    # x[3] is the time in day.
    # x[4] is the grid block porosity.
    # x[5] is the grid block x-permeability in mD.   
   
    # Model Outputs:
    # out[0] is the predicted grid block pressure (psia).
    # out[1] is the predicted grid block gas saturation.
    # out[2] is the predicted grid block oil saturation.
    
    # Training Labels: 
    # y[0] is the training label grid block pressure (psia).
    # y[1] is the training label block saturation--gas.
    # y[2] is the training label block saturation--oil.
    
    with tf.GradientTape(persistent=True) as tape3:
        pinn_errors,outs,checks,no_blks=model.loss_func['Physics_Error'](model,x,y) 
        dom_pinn=pinn_errors[0]
        dbc_pinn=pinn_errors[1]
        nbc_pinn=pinn_errors[2]
        ibc_pinn=pinn_errors[3]
        ic_pinn=pinn_errors[4]
        qrc_pinn=tf.stack(pinn_errors[5:])         
        mbc_pinn=checks[0]                          # MBC: Tank Material Balance Check.
        cmbc_pinn=checks[1]                         # CMBC: Cumulative Tank Material Balance Check.
        # =============================================== Training Data ============================================================
        # Training data includes the pressure and gas saturation labels.

        y_label=tf.stack([model.loss_func['Reshape'](y[i]) for i in model.nT_list],axis=0)
        y_model=outs[0][0:model.nT]
    
        #y_label=[dnn.normalization_Layer(stat_limits=model.ts[7:8,:2][0],norm_limits=[0,1],norm_type='linear',invert=False,layer_name='')(l) for l in y_label]
        #y_model=[dnn.normalization_Layer(stat_limits=model.ts[7:8+i,:2][0],norm_limits=[0,1],norm_type='linear',invert=False,layer_name='')(y_model[i]) for i in range(len(y_model))]

        #q_well_idx=tf.reshape(tf.expand_dims(tf.scatter_nd(model.cfd_type['Conn_Idx'], tf.ones_like(model.cfd_type['Init_Grate']), model.cfd_type['Dimension']['Reshape'][1:]),0)*tf.ones_like(y_model),tf.shape(y_model)[1:])

        td=(y_label-y_model)
        # Calculate the (Euclidean norm)**2 of each solution term--i.e., the error term.
        dom_pinn_se=tf.math.square(dom_pinn)                 
        dbc_pinn_se=tf.math.square(dbc_pinn)
        nbc_pinn_se=tf.math.square(nbc_pinn) 
        ibc_pinn_se=tf.math.square(ibc_pinn)
        ic_pinn_se=tf.math.square(ic_pinn) 
        qrc_pinn_se=tf.math.square(qrc_pinn)
        mbc_pinn_se=tf.math.square(mbc_pinn)
        cmbc_pinn_se=tf.math.square(cmbc_pinn)  
        
        # Calculate the (Euclidean norm)**2 of the training data term.
        td_se=tf.math.square(td)
       
        # Compute the Sum of Squared Errors (SSE). 
        dom_pinn_sse=tf.math.reduce_sum(dom_pinn_se)
        dbc_pinn_sse=tf.math.reduce_sum(dbc_pinn_se)
        nbc_pinn_sse=tf.math.reduce_sum(nbc_pinn_se)
        ibc_pinn_sse=tf.math.reduce_sum(ibc_pinn_se)
        ic_pinn_sse=tf.math.reduce_sum(ic_pinn_se)
        qrc_pinn_sse=tf.math.reduce_sum(qrc_pinn_se)
        mbc_pinn_sse=tf.math.reduce_sum(mbc_pinn_se)
        cmbc_pinn_sse=tf.math.reduce_sum(cmbc_pinn_se)

        # Compute the Sum of Squared Errors (SSE) of the training data term.
        td_sse=tf.math.reduce_sum(td_se,axis=model.loss_func['Reduce_Axis'])
        
        # Weight the regularization term. 
        dom_wsse=model.nwt[0]*dom_pinn_sse
        dbc_wsse=model.nwt[1]*dbc_pinn_sse
        nbc_wsse=model.nwt[2]*(nbc_pinn_sse+tf.reduce_sum(qrc_pinn_sse)) #Rate check is averaged with the NBC Loss.              # nbc_avg_pinn_sse
        ibc_wsse=model.nwt[3]*ibc_pinn_sse
        ic_wsse=model.nwt[4]*ic_pinn_sse                      
        mbc_wsse=model.nwt[5]*mbc_pinn_sse
        cmbc_wsse=model.nwt[6]*cmbc_pinn_sse
        
        # Compute the weighted training loss of the batch. Also normalized using the training mean. 
        td_wsse=model.nwt[7:(7+model.nT)]*td_sse
        batch_wsse = dom_wsse+dbc_wsse+nbc_wsse+ibc_wsse+ic_wsse+mbc_wsse+cmbc_wsse+tf.reduce_sum(td_wsse)

        # Count the unique appearance of each loss term that does not have a zero identifier
        dom_error_count=tf.math.reduce_sum(tf.ones_like(dom_pinn_se))
        dbc_error_count=tf.math.reduce_sum(tf.ones_like(dbc_pinn_se))
        nbc_error_count=tf.math.reduce_sum(tf.ones_like(nbc_pinn_se))
        ibc_error_count=tf.math.reduce_sum(tf.ones_like(ibc_pinn_se))
        ic_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))
        mbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))                   # Tank Model 
        cmbc_error_count=tf.math.reduce_sum(tf.ones_like(ic_pinn_se))  
        td_error_count=tf.math.reduce_sum(tf.ones_like(td_se[0]))
                
        # Compute the batch Mean Squared Errors (MSE)--for reporting purpose only
        dom_wmse=dom_wsse/zeros_to_ones(dom_error_count)
        dbc_wmse=dbc_wsse/zeros_to_ones(dbc_error_count)
        nbc_wmse=nbc_wsse/zeros_to_ones(nbc_error_count)
        ibc_wmse=ibc_wsse/zeros_to_ones(ibc_error_count)
        ic_wmse=ic_wsse/zeros_to_ones(ic_error_count)
        mbc_wmse=mbc_wsse/zeros_to_ones(mbc_error_count)
        cmbc_wmse=cmbc_wsse/zeros_to_ones(cmbc_error_count)
        td_wmse=td_wsse/zeros_to_ones(td_error_count)

        # tf.print('DOM_WMSE\n',dom_wsse,'\nDBC_WMSE\n',dbc_wsse,'\nNBC_WMSE\n',nbc_wsse,'\nIBC_WMSE\n',ibc_wsse,'\nIC_WMSE\n',ic_wsse,'\nMBC_WMSE\n',mbc_wsse,'\nCMBC_WMSE\n',cmbc_wsse,'\nTD_WMSE\n',td_wmse,output_stream="file://C:/Users/VCM1/Documents/PHD_HW_Machine_Learning/ML_Cases/Physics/Dry_Gas/debug.out" )
        batch_wmse = dom_wmse+dbc_wmse+nbc_wmse+ibc_wmse+ic_wmse+mbc_wmse+cmbc_wmse+tf.reduce_sum(td_wmse)                # td_see is reduced as it's a matrix.
             
    # Compute the gradients of each loss term.
    dom_wsse_grad=tape3.gradient(dom_wsse, model.trainable_variables,unconnected_gradients='zero')
    dbc_wsse_grad=tape3.gradient(dbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    nbc_wsse_grad=tape3.gradient(nbc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ibc_wsse_grad=tape3.gradient(ibc_wsse, model.trainable_variables,unconnected_gradients='zero')
    ic_wsse_grad=tape3.gradient(ic_wsse, model.trainable_variables,unconnected_gradients='zero')
    mbc_wsse_grad=tape3.gradient(mbc_wsse, model.trainable_variables,unconnected_gradients='zero')   
    cmbc_wsse_grad=tape3.gradient(cmbc_wsse, model.trainable_variables,unconnected_gradients='zero') 
    td_wsse_grad=tape3.gradient(td_wsse, model.trainable_variables,unconnected_gradients='zero')      # Gradient for the training data has more than one column--constitutive relationship. QoIs etc.

    # Compute the gradient of the batched data.
    batch_wsse_grad=tape3.gradient(batch_wsse, model.trainable_variables,unconnected_gradients='zero')
    del tape3
    
    # Arrange the variables as a list. 
    _wsse=[batch_wsse,dom_wsse,dbc_wsse,nbc_wsse,ibc_wsse,ic_wsse,mbc_wsse,cmbc_wsse,(td_wsse)]
    _wsse_grad=[batch_wsse_grad,dom_wsse_grad,dbc_wsse_grad,nbc_wsse_grad,ibc_wsse_grad,ic_wsse_grad,mbc_wsse_grad,cmbc_wsse_grad,td_wsse_grad]
    error_count=[1,dom_error_count,dbc_error_count,nbc_error_count,ibc_error_count,ic_error_count,mbc_error_count,cmbc_error_count,tf.reduce_sum(td_error_count)]

    _wmse=[batch_wmse,dom_wmse,dbc_wmse,nbc_wmse,ibc_wmse,ic_wmse,mbc_wmse,cmbc_wmse,td_wmse] 

    #return [_wsse,_wsse_grad,error_count,_wmse,model.loss_func['Squeeze_Out'](tf.reshape(outs[0][0:model.nT,...],(model.nT,-1,*model.cfd_type['Dimension']['Dim'])))]
    return [_wsse,_wsse_grad,error_count,_wmse,outs[0][0:model.nT,...]]


