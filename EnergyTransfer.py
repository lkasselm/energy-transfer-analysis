import numpy as np
import FFTHelperFuncs
from mpi4py_fft import newDistArray
from MPIderivHelperFuncs import MPIderiv2, MPIXdotGradYScalar, MPIXdotGradY, MPIdivX, MPIdivXY, MPIgradX
import time
import pickle
import sys

class EnergyTransfer:
    
    
    def __init__(self, MPI, RES, fields, gamma, box_length):
        
        self.gamma = gamma
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.RES = RES

        # Load fields and convert to units where the box has a linear size of 1. 
        self.L = box_length 
        self.rho = fields['rho'] 
        self.U = fields['U'] 
        self.B = fields['B'] 
        self.Acc = fields['Acc']
        self.P = fields['P']  

        # Variables that we might (or might not) use later depending on the different definitons of terms
        self.W = None
        self.FT_W = None
        self.S = None
        self.FT_S = None
        self.FT_B = None
        self.FT_Acc = None
        self.FT_P = None
        self.FT_rho = None
        self.FT_U = None

        self.FFT = FFTHelperFuncs.FFT
        self.localKmag = np.linalg.norm(FFTHelperFuncs.local_wavenumbermesh,axis=0)
        self.localK = FFTHelperFuncs.local_wavenumbermesh 
    
    def convert_to_physical_units(self, transfer_term):
        """ convert transfer term to physical units 
        
            (The code assumes a box size of 1 for the gradient definition and for the integral
            it just sums over all grid points. To get the physical units, multiply integral by 
            cell volume L^3 / RES^3 and gradients by 1/L. --> overall factor L^2 / RES^3)
        """
        return transfer_term * self.L**2 / (self.RES**3)

    def getShellX(self,FTquant,Low,Up):
        """ extracts shell X-0.5 < K <X+0.5 of FTquant """

        if FTquant.shape[0] == 3:    
            Quant_X = newDistArray(self.FFT,False,rank=1)
            for i in range(3):
                tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant[i],0.)
                Quant_X[i] = self.FFT.backward(tmp,Quant_X[i])
        else:
            Quant_X = newDistArray(self.FFT,False)
            tmp = np.where(np.logical_and(self.localKmag > Low, self.localKmag <= Up),FTquant,0.)
            Quant_X = self.FFT.backward(tmp,Quant_X)        

        return Quant_X
    
    def Helicity_decomp(self, FTquant):
        """ perform helicity decomposition of vector field FTquant """

        if FTquant.shape[0] != 3:
            raise SystemExit("Helicity decomposition only implemented for vector fields")
    
        # construct helical basis:
        kx = self.localK[0]
        ky = self.localK[1]
        kz = self.localK[2]
        k_mag = self.localKmag
        
        # find vector orthogonal to k
        


        

    
    def populateResultDict(self,Result,KBins,formalism,Terms,method):
        if self.comm.Get_rank() != 0:
            return
            
        if formalism not in Result.keys():
            Result[formalism] = {}
        
        for term in Terms:
            if term not in Result[formalism].keys():
                Result[formalism][term] = {}
        
            
            if method not in Result[formalism][term].keys():
                Result[formalism][term][method] = {}
                            

            for i in range(len(KBins)-1):
                KBin = "%.2f-%.2f" % (KBins[i],KBins[i+1])
                if KBin not in Result[formalism][term][method].keys():
                    Result[formalism][term][method][KBin] = {}              

    def addResultToDict(self,Result,formalism,term,method,KBin,QBin,value):

        value = self.convert_to_physical_units(value)

        if self.comm.Get_rank() != 0:
            return
            
        if formalism not in Result.keys():
            Result[formalism] = {}
        

        if term not in Result[formalism].keys():
            Result[formalism][term] = {}


        if method not in Result[formalism][term].keys():
            Result[formalism][term][method] = {}

        if KBin not in Result[formalism][term][method].keys():
            Result[formalism][term][method][KBin] = {}
                    
        Result[formalism][term][method][KBin][QBin] = float(value)
                    
    def calcBasicVars(self,formalism):
        """ calculate basic variables for the different formalisms, i.e.
        W and FT_W for WW formalism, and
        """
        
        rho = self.rho
        P = self.P
        U = self.U
        B = self.B

        if self.W is None:
            self.W = newDistArray(self.FFT,False,rank=1)                                
            for i in range(3):
                self.W[i] = np.sqrt(rho) * U[i]

        if self.S is None and P is not None:
            self.S = np.sqrt(self.gamma*P)

        if self.FT_W is None:
            self.FT_W = newDistArray(self.FFT,rank=1)
            for i in range(3):
                self.FT_W[i] = self.FFT.forward(self.W[i], self.FT_W[i])            
            
        if self.FT_B is None and self.B is not None:
            self.FT_B = newDistArray(self.FFT,rank=1)
            for i in range(3):
                self.FT_B[i] = self.FFT.forward(self.B[i], self.FT_B[i])    
        
        if self.FT_P is None and self.P is not None:
            self.FT_P = newDistArray(self.FFT)
            self.FT_P = self.FFT.forward(self.P, self.FT_P)    
        
        if self.FT_S is None and self.S is not None:
            self.FT_S = newDistArray(self.FFT)
            self.FT_S = self.FFT.forward(self.S, self.FT_S)    
        
        if self.FT_Acc is None and self.Acc is not None:
            self.FT_Acc = newDistArray(self.FFT,rank=1)
            for i in range(3):
                self.FT_Acc[i] = self.FFT.forward(self.Acc[i], self.FT_Acc[i])    
            
    
    def getTransferWWAnyToAny(self, Result, KBins, QBins, Terms):
        """ return what
                    formalism -- determined by the definiton of the spectral kinetic energy density
                "WW": E_kin(k) = 1/2 |FT(sqrt(rho)U)|^2
        
        Args:
            Result -- a (potentially empty) dictionary to store the results in
            Ks -- range of destination shell wavenumber
            Qs -- range of source shell wavenumbers
            Terms -- list of terms that should be analyzed, could be
                "UUA": Kinetic to kinetic by advection        
        """
        
        #self.populateResultDict(Result,KBins,"WW",Terms,"AnyToAny")
        self.calcBasicVars("WW")
        
        
        rho = self.rho
        U = self.U
        B = self.B
        S = self.S
        W = self.W
        FT_W = self.FT_W
        FT_S = self.FT_S
        FT_B = self.FT_B
        FT_P = self.FT_P
        FT_Acc = self.FT_Acc

        startTime = time.time()

        # clear Q terms
        W_Q = None
        S_Q = None
        B_Q = None
        SDivW_QoverGammaSqrtRho = None
        OneOverGammaSqrtRhogradSS_Q = None
        OneOverTwoSqrtRhogradBB_Q = None
        UdotGradW_Q = None
        UdotGradS_Q = None
        UdotGradB_Q = None
        bDotGradB_Q = None
        BdotGradW_QoverSqrtRho = None
        DivbW_Q = None
        bdotGradW_Q = None
        W_QoverSqrtRho = None
        W_QoverSqrtRhoDotGradB = None
        DivW_QoverSqrtRho = None        
        DivW_Qb = None
        W_QdotGradb = None
        DivW_Q = None
        BDivW_Qover2SqrtRho = None
        OneOverSqrtRhoGradP_Q = None
        SqrtRhoAcc_Q = None
        
        DivU = None
        b = None
        Divb = None
        
        for q in range(len(QBins)-1):
            QBin = "%.2f-%.2f" % (QBins[q],QBins[q+1])

            # clear K terms
            W_K = None
            S_K = None	
            B_K = None
            
            for k in range(len(KBins)-1):
                
                KBin = "%.2f-%.2f" % (KBins[k],KBins[k+1])

                #  - W_K * (U dot grad) W_Q - 0.5 W_K W_Q DivU
                if "UU" in Terms:
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                        
                    if UdotGradW_Q is None:
                        UdotGradW_Q = MPIXdotGradY(self.comm,U,W_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - np.sum(W_K * UdotGradW_Q)              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * W_K * W_Q * DivU)                    

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UUA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","UUC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UU","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))   
                
                #  - S_K * (U dot grad) S_Q - 0.5 S_K S_Q DivU
                if "SS" in Terms:
                    if S_K is None:
                        S_K = self.getShellX(FT_S,KBins[k],KBins[k+1])
                    
                    if S_Q is None:
                        S_Q = self.getShellX(FT_S,QBins[q],QBins[q+1])                        
                        
                    if UdotGradS_Q is None:
                        UdotGradS_Q = MPIXdotGradYScalar(self.comm,U,S_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - 2./self.gamma/(self.gamma - 1.) * np.sum(S_K * UdotGradS_Q)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - 1./self.gamma/(self.gamma - 1.) * np.sum(S_K * S_Q * DivU)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","SSA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","SSC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","SS","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with SS for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))   
                
                if "BB" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])                        
                        
                    if UdotGradB_Q is None:
                        UdotGradB_Q = MPIXdotGradY(self.comm,U,B_Q)                        
                    
                    if DivU is None:
                        DivU = MPIdivX(self.comm,U)
                    
                    
                    localSum = - np.sum(B_K * UdotGradB_Q) # Advective              

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    localSum = - np.sum(0.5 * B_K * B_Q * DivU) # Compressive                    

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                    
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BBA","AnyToAny",KBin,QBin,totalSumA)
                        self.addResultToDict(Result,"WW","BBC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","BB","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with BB for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                

                # W_K * (1/sqrt(rho) B dot grad) B_Q
                if "BUT" in Terms:
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                        
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                    
                    if b is None:
                        b = B/np.sqrt(rho) # Alfven velocity
                        
                    if bDotGradB_Q is None:
                        bDotGradB_Q = MPIXdotGradY(self.comm,b,B_Q)                        
                        
                   
                    localSum = np.sum(W_K * bDotGradB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUT","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUT for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                
                if "UBT" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                                          
                        
                    if BdotGradW_QoverSqrtRho is None:
                        BdotGradW_QoverSqrtRho = MPIXdotGradY(self.comm,B,W_Q/np.sqrt(rho))    
     
                    # B_K * (B dot grad) W_Q/sqrt(rho) - Moss et al
                    localSum = np.sum(B_K * BdotGradW_QoverSqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTa","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBTa for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                        
                        
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if DivbW_Q is None:
                        DivbW_Q = MPIdivXY(self.comm,b,W_Q)
                        
                    # B^K_i pd_j b_j W^Q_i - total term
                    localSum = np.sum(B_K * DivbW_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBTb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                        
                    # B^K_i  b_j pd_j W^Q_i - "adv" term                    
                    if bdotGradW_Q is None:
                        bdotGradW_Q = MPIXdotGradY(self.comm,b,W_Q) 
                        
                    localSum = np.sum(B_K * bdotGradW_Q)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTbA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBTbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                        
                    # B^K_i  W^Q_i  pd_j b_j - "compr" term                    
                    if Divb is None:
                        Divb = MPIdivX(self.comm,b)
                        
                    localSum = np.sum(B_K * W_Q *Divb)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBTbC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBTbTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBTbC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                        


                # B * (1/sqrt(rho) W_K dot grad) B_Q
                if "BUP" in Terms:
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                    
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                        
                    W_KDotGradB_Q = MPIXdotGradY(self.comm,W_K,B_Q)
                    
                    localSum = - np.sum(b * W_KDotGradB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUP","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUP for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # this is the term with split BB
                if "BUPbb" in Terms:
                    
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                        
                    if OneOverTwoSqrtRhogradBB_Q is None:
                        OneOverTwoSqrtRhogradBB_Q = MPIgradX(self.comm, np.sum(B * B_Q,axis=0))/ (2. * np.sqrt(rho))
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                                            
                    
                    localSum = - np.sum(W_K * OneOverTwoSqrtRhogradBB_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","BUPbb","AnyToAny",KBin,QBin,totalSum)
                        print("done with BUPbb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                
                if "UBPa" in Terms:
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                    
                    if W_QoverSqrtRho is None:
                        W_QoverSqrtRho = W_Q/np.sqrt(rho)
                    
                    if W_QoverSqrtRhoDotGradB is None:
                        W_QoverSqrtRhoDotGradB = MPIXdotGradY(self.comm,W_QoverSqrtRho,B)
                    
                    localSum = - np.sum(B_K * W_QoverSqrtRhoDotGradB)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPaA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBPaA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                    if DivW_QoverSqrtRho is None:
                        DivW_QoverSqrtRho = MPIdivX(self.comm,W_QoverSqrtRho) 
    
                    
                    localSum = - np.sum(B_K * B * DivW_QoverSqrtRho)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPaC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBPaTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBPaC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))    

                if "UBPbb" in Terms:

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])  
                        
                    if BDivW_Qover2SqrtRho is None:
                        BDivW_Qover2SqrtRho = B * MPIdivX(self.comm, W_Q/np.sqrt(rho)/2. )                         
                        
                    localSum = - np.sum(B_K * BDivW_Qover2SqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBPbb for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        
                if "UBPb" in Terms:

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])                        
                    
                    if b is None:
                        b = B/np.sqrt(rho)
                        
                    if DivW_Qb is None:
                        DivW_Qb = MPIdivXY(self.comm,W_Q,b)
                        
                    localSum = - np.sum(B_K * DivW_Qb)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPb","AnyToAny",KBin,QBin,totalSum)
                        print("done with UBPbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
            
                    
                    if W_QdotGradb is None:
                        W_QdotGradb = MPIXdotGradY(self.comm,W_Q,b)
                    
                    localSum = - np.sum(B_K * W_QdotGradb)

                    totalSumA = None
                    totalSumA = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbA","AnyToAny",KBin,QBin,totalSumA)
                        print("done with UBPbA for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                    if DivW_Q is None:
                        DivW_Q = MPIdivX(self.comm,W_Q) 
    
                    
                    localSum = - np.sum(B_K * B * DivW_Q)

                    totalSumB = None
                    totalSumB = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","UBPbC","AnyToAny",KBin,QBin,totalSumB)
                        self.addResultToDict(Result,"WW","UBPbTot","AnyToAny",KBin,QBin,totalSumA+totalSumB)
                        print("done with UBPbC for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))    
                
                
                if "SU" in Terms:
                    
                    if S_Q is None:
                        S_Q = self.getShellX(FT_S,QBins[q],QBins[q+1])
                        
                    # TODO reuse vars here with BUP terms
                    if OneOverGammaSqrtRhogradSS_Q is None:
                        OneOverGammaSqrtRhogradSS_Q = MPIgradX(self.comm, (S * S_Q))/ (self.gamma * np.sqrt(rho))
                        
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                                            
                    
                    localSum = - np.sum(W_K * OneOverGammaSqrtRhogradSS_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","SU","AnyToAny",KBin,QBin,totalSum)
                        print("done with SU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                if "US" in Terms:

                    if S_K is None:
                        S_K = self.getShellX(FT_S,KBins[k],KBins[k+1])                      
                        
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])  
                    # TODO reuse vars here with BUP terms
                    if SDivW_QoverGammaSqrtRho is None:
                        SDivW_QoverGammaSqrtRho = S * MPIdivX(self.comm, W_Q/np.sqrt(rho)/self.gamma )                         
                        
                    localSum = - np.sum(S_K * SDivW_QoverGammaSqrtRho)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)                        
                        
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","US","AnyToAny",KBin,QBin,totalSum)
                        print("done with US for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))                        

                # - W_K 1/sqrt(rho) grad P_Q
                if "PU" in Terms:

                    if OneOverSqrtRhoGradP_Q is None:
                        P_Q = self.getShellX(FT_P,QBins[q],QBins[q+1])
                        OneOverSqrtRhoGradP_Q = MPIgradX(self.comm, P_Q)/ np.sqrt(rho)
                        del P_Q
                    
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    localSum = - np.sum(W_K * OneOverSqrtRhoGradP_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","PU","AnyToAny",KBin,QBin,totalSum)
                        print("done with PU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))
                
                # W_K sqrt(rho) Acc_Q
                if "FU" in Terms:

                    if SqrtRhoAcc_Q is None:
                        SqrtRhoAcc_Q = np.sqrt(rho) * self.getShellX(FT_Acc,QBins[q],QBins[q+1])
                    
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    localSum = np.sum(W_K * SqrtRhoAcc_Q)

                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)
                    
                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","FU","AnyToAny",KBin,QBin,totalSum)
                        print("done with FU for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # Helicity transfer: 
                if "H" in Terms:
                    # 2 * (B_k * (U x B_q)), e.g. doi:10.1017/jfm.2021.496 equation (4.1)

                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])

                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])

                    localSum = 2. * np.sum(B_K * np.cross(U, B_Q, axis=0))
                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","H","AnyToAny",KBin,QBin,totalSum)
                        print("done with H for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                if "T" in Terms:
                    # Total triadic coupeling leading to energy increase in specific I shell (here I = 10), I.e. T_{Ikq}

                    B_I = self.getShellX(FT_B,KBins[7],KBins[8])
                    if B_K is None:
                        B_K = self.getShellX(FT_B,KBins[k],KBins[k+1])
                    if W_Q is None:
                        W_Q = self.getShellX(FT_W,QBins[q],QBins[q+1])
                    if B_Q is None:
                        B_Q = self.getShellX(FT_B,QBins[q],QBins[q+1])
                    if W_K is None:
                        W_K = self.getShellX(FT_W,KBins[k],KBins[k+1])
                    
                    # B_I dot (B_Q dot grad) W_K
                    B_QdotGradB_I = MPIXdotGradY(self.comm,B_Q,W_K)
                    localSum = np.sum(B_I * B_QdotGradB_I)
                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","TUBT","AnyToAny",KBin,QBin,totalSum)
                    
                    # - B_I dot (W_K dot grad) B_Q
                    W_KdotGradB_Q = MPIXdotGradY(self.comm,W_K,B_Q)
                    localSum = - np.sum(B_I * W_KdotGradB_Q)
                    totalSum = None
                    totalSum = self.comm.reduce(sendobj=localSum, op=self.MPI.SUM, root=0)

                    if self.comm.Get_rank() == 0:
                        self.addResultToDict(Result,"WW","TBB","AnyToAny",KBin,QBin,totalSum)
                        print("done with T for K = %s Q = %s after %.1f sec [total]" % (KBin,QBin,time.time() - startTime ))

                # clear K terms
                W_K = None
                S_K = None
                B_K = None
        

            # clear Q terms
            W_Q = None
            S_Q = None
            B_Q = None
            OneOverTwoSqrtRhogradBB_Q = None
            SDivW_QoverGammaSqrtRho  = None
            OneOverGammaSqrtRhogradSS_Q = None
            UdotGradW_Q = None
            UdotGradS_Q = None
            UdotGradB_Q = None
            bDotGradB_Q = None
            BdotGradW_QoverSqrtRho = None
            DivbW_Q = None
            bdotGradW_Q = None
            W_QoverSqrtRho = None
            W_QoverSqrtRhoDotGradB = None
            DivW_QoverSqrtRho = None
            DivW_Qb = None
            W_QdotGradb = None
            DivW_Q = None
            BDivW_Qover2SqrtRho = None
            OneOverSqrtRhoGradP_Q = None
            SqrtRhoAcc_Q = None
            
            Str = ""
            for Term in Terms:
                Str += "-" + Term            
            if self.comm.Get_rank() != 0 and False:
                pickle.dump(Result,open("tmp%s.pkl" % Str,"wb")) 
