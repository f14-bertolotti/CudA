from math import exp
import numpy as np
from torch.fft import fft2, ifft2
import matplotlib.pyplot as plt
import torch
np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True, precision=3)


orbium = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18620394164714388, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03803717242518284, 0.06245837527728269, 0.04988763233949314, 0.3320774385558396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03773608099239814, 0.11957704919278458, 0.15802368862792687, 0.16330556820620415, 0.16974424951181388, 0.21343988496118207, 0.48639733975738747, 0.3297301741915967, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.015120697883400425, 0.06577791518220094, 0.09392484720152754, 0.0812589544929557, 0.04018975643552394, 0.14142816323103613, 0.15842620775483066, 0.0771833422814457, 0.0031988431039820854, 0.0, 0.0, 0.024856352242369553, 0.30786022916327216, 0.6740534174957, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.045456920211471005, 0.1667336025850817, 0.2623852102145015, 0.31259098376557365, 0.29651185031854893, 0.16594676405744463, 0.0706709947806405, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6155752014980821, 0.0, 0.0, 0.0, 0.0], [0.0, 0.009676009186942558, 0.21136313511224705, 0.3646668998034019, 0.4649448754920916, 0.4440897207769938, 0.24389633063803584, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.41880473353962794, 0.01581828347059655, 0.0, 0.0], [0.0, 0.0, 0.1797360486547171, 0.38529083526367625, 0.489866917007241, 0.42371288199297763, 0.26256925153395466, 0.1611574412130361, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03330983074689085, 0.3034904134264006, 0.0, 0.0], [0.0, 0.0, 0.09707271216871541, 0.3092981352722303, 0.36021609823076334, 0.2942380512480376, 0.2640286341674995, 0.2955839010094476, 0.34271652075340825, 0.2919178209368468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1401238149559207, 0.1842408314139363, 0.0], [0.0, 0.05137419793807378, 0.16095271972816405, 0.14636787334981072, 0.03794901994003126, 0.1670596604946225, 0.2856044620387996, 0.40761279642573345, 0.5181288137781067, 0.5924838764492517, 0.6261959088253283, 0.2789710190509281, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03214827686379437, 0.20308003049579618, 0.02904297799462934], [0.0, 0.10103760708412172, 0.15960823487620174, 0.04335728833639305, 0.0, 0.0, 0.28656077047856193, 0.48442675003303026, 0.6545608039329831, 0.7797639437006925, 0.8607172787224697, 0.887757958737377, 0.6138818589273715, 0.024226890211743332, 0.0, 0.0, 0.0, 0.01821297386195453, 0.17717971556345824, 0.10366698197998506], [0.0, 0.148260787282635, 0.09583549191033583, 0.0, 0.0, 0.0, 0.06278144781939668, 0.5065531283016997, 0.7255056054109946, 0.8935967764080974, 0.9711523589314585, 1.0, 1.0, 0.7837901419806675, 0.20132113925609743, 0.0, 0.0, 0.060005102737890836, 0.17558681062455972, 0.14548764560795002], [0.0, 0.1578681785163286, 0.07321953091976355, 0.0, 0.0, 0.0, 0.0, 0.2889237542909928, 0.74455963243658, 0.9523526645019897, 0.9742045150084269, 0.8860658433600919, 0.9800126220091733, 1.0, 0.7209939678210999, 0.3021559485619194, 0.1362206222385685, 0.14607258600948686, 0.20365746219257877, 0.1509599112909949], [0.0, 0.1438666658377839, 0.09382359707793175, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4710142824664052, 0.9541901180271966, 1.0, 0.8604778161932515, 0.7945356246448013, 0.8848830449888537, 0.8536219219111439, 0.5805656450703742, 0.34371511590379544, 0.2564690715996305, 0.2321081259402155, 0.13562836569873715], [0.28999041013158805, 0.0643810022859383, 0.19228321805656234, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5307307518032851, 1.0, 1.0, 0.8543161338206895, 0.8256553935981522, 0.8015273999967775, 0.6541124679877278, 0.45543506999013084, 0.3299349836458293, 0.23851837081441646, 0.09543933134255442], [0.0, 0.15507996628940807, 0.48067336186065135, 0.03640929295787496, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.46574071314021065, 0.8814152336656682, 0.8958699408691204, 0.8123992758353427, 0.7269544300185717, 0.6016020095440573, 0.4495161196438381, 0.3227349443245773, 0.19786472150503226, 0.053340755361904794], [0.0, 0.0, 0.037605209814933704, 0.6707065737980381, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35774565785544493, 0.6275058409358334, 0.6755094874784631, 0.6102782654128045, 0.5050768408063153, 0.3808711926578331, 0.2544249444090202, 0.1312300032666778, 0.009989421725527525], [0.0, 0.0, 0.0, 0.0, 0.6558014901715007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09121184105072119, 0.3048865178933467, 0.4458360821914249, 0.4581648585565114, 0.3888641659950559, 0.28411381470755753, 0.16431882856293323, 0.052785283646939046, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5199703759401852, 0.08755772428922252, 0.0, 0.0, 0.0, 0.0, 0.06587573846962441, 0.20441290332385775, 0.30441378998290974, 0.3212173872391406, 0.2697645688777056, 0.175541686235464, 0.07076336763352298, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30190367592388173, 0.25333206348438564, 0.1163347081995864, 0.07405490216418129, 0.09327124988254325, 0.1423816543532342, 0.20213258481510016, 0.2433289019905386, 0.22848383770726666, 0.15459181660131205, 0.06129244097956792, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.046151287480330844, 0.18251568520758568, 0.2078606187002749, 0.1980219538686351, 0.19450055978431297, 0.17879884911526855, 0.15368233874298384, 0.08395809738797887, 0.028720937802170467, 0.0, 0.0, 0.0, 0.0]]

def compile_cpp(x)->str:
    cppstr = ""
    for i,r in enumerate(x):
        for j,v in enumerate(r):
            if v != 0 : cppstr += f"ptr[{i}*size+{j}]={v:.3f}f;"
    return cppstr


class Lenia(torch.nn.Module):

    def __init__(self, size=128, kernel_size=13, mu=0.5, gmu=0.135, sigma=0.15, gsigma=0.015, dtime=0.1, device="cpu"):
        super().__init__()
        self.size        = size
        self.kernel_size = kernel_size
        self.mu          = mu
        self.gmu         = gmu
        self.sigma       = sigma
        self.gsigma      = gsigma
        self.dtime       = dtime
        #self.kernel      = torch.zeros(size, size, dtype=torch.float, device=device)

        #for i in range(-self.kernel_size//2, self.kernel_size//2, 1):
        #    for j in range(-self.kernel_size//2, self.kernel_size//2, 1):
        #        norm = (i**2 + j**2)**.5 / (self.kernel_size // 2);
        #        value = (norm < 1) * exp(-((norm - self.mu)/self.sigma)**2 / 2);
        #        self.kernel[i,j] = value

        #self.kernel /= self.kernel.sum()
        #self.kernel = rfft2(self.kernel)

        D = np.linalg.norm(np.array(np.ogrid[-self.size//2:self.size//2, -self.size//2:self.size//2], dtype=object)) / self.kernel_size
        K = (D<1) * np.exp(-((D-self.mu)/self.sigma)**2 / 2) 
        fK = np.fft.fft2(np.fft.fftshift(K / np.sum(K)))
        self.kernel = torch.tensor(fK, device=device).unsqueeze(0)

    def growth(self, x):
        return torch.exp(-((x-self.gmu)/self.gsigma)**2 / 2) * 2  - 1

    def run(self, grid):
        return torch.clip(grid + self.dtime * self.growth(ifft2(self.kernel * fft2(grid)).to(float)),0,1)

    def show(self, grid) -> None:
        plt.imshow(grid.detach().cpu().numpy())
        plt.show()

class Model1(torch.nn.Module):
    def __init__(self, batch_size = 10, iterations = 41, device="cpu"):
        super().__init__()
        self.device = device
        self.lenia = Lenia(size=64, device=device)
        self.iterations = iterations

        self.mu_lin0 = torch.nn.Linear(1,400, device=device)
        self.mu_lin1 = torch.nn.Linear(400,400, device=device)
        self.mu_lin2 = torch.nn.Linear(400,400, device=device)
        self.mu_lin3 = torch.nn.Linear(400,400, device=device)
        self.mu_ln0 = torch.nn.LayerNorm(400, device=device)
        self.mu_ln1 = torch.nn.LayerNorm(400, device=device)
        self.mu_ln2 = torch.nn.LayerNorm(400, device=device)
        self.sigma_lin0 = torch.nn.Linear(1,400, device=device)
        self.sigma_lin1 = torch.nn.Linear(400,400, device=device)
        self.sigma_lin2 = torch.nn.Linear(400,400, device=device)
        self.sigma_lin3 = torch.nn.Linear(400,400, device=device)
        self.sigma_ln0 = torch.nn.LayerNorm(400, device=device)
        self.sigma_ln1 = torch.nn.LayerNorm(400, device=device)
        self.sigma_ln2 = torch.nn.LayerNorm(400, device=device)

        self.mean = torch.zeros(batch_size, 20,20).to(device)
        self.std  = torch. ones(batch_size, 20,20).to(device)  

        self.mu = torch.nn.Parameter(torch.rand((20,20), requires_grad=True).to(device))
        self.sigma = torch.nn.Parameter(torch.rand((20,20), requires_grad=True).to(device))



    def forward(self): 

        mu = self.mu_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        mu = self.mu_ln0(mu + torch.nn.functional.relu(self.mu_lin1(mu)))
        mu = self.mu_ln1(mu + torch.nn.functional.relu(self.mu_lin2(mu)))
        mu = self.mu_ln2(mu + torch.nn.functional.relu(self.mu_lin3(mu)))
        mu = mu.reshape(20,20)
        sigma = self.sigma_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        sigma = self.sigma_ln0(sigma + torch.nn.functional.relu(self.sigma_lin1(sigma)))
        sigma = self.sigma_ln1(sigma + torch.nn.functional.relu(self.sigma_lin2(sigma)))
        sigma = self.sigma_ln2(sigma + torch.nn.functional.relu(self.sigma_lin3(sigma)))
        sigma = sigma.reshape(20,20)


        x = torch.normal(mean=self.mean, std=self.std) * sigma + mu
        x = torch.clamp(x, 0, 1)
        x = torch.nn.functional.pad(x, (0,44,0,44,0,0))
        signal = mid_signal = running_signal = x


        loss = torch.zeros(signal.size(0)).to(signal.device);
        for i in range(1,self.iterations): 
            if i == self.iterations // 2: mid_signal = running_signal
            running_signal = self.lenia.run(running_signal)  
            loss += running_signal[:,20:].mean(2).mean(1)
            loss += ((signal[:,:20,:20] - running_signal[:,int(i//1.5):int(20+i//1.5),int(i//1.5):int(20+i//1.5)])**2).mean(2).mean(1)
            loss -= running_signal[:,int(i//1.5):int(20+i//1.5),int(i//1.5):int(20+i//1.5)].mean(2).mean(1)

        return loss, signal, mid_signal, running_signal


class Model2(torch.nn.Module):
    def __init__(self, batch_size = 10, iterations = 41, device="cpu"):
        super().__init__()
        self.device = device
        self.lenia = Lenia(size=64, device=device)
        self.iterations = iterations

        self.mu_lin0 = torch.nn.Linear(1,2500, device=device)
        self.mu_lin1 = torch.nn.Linear(2500,2500, device=device)
        self.mu_lin2 = torch.nn.Linear(2500,2500, device=device)
        self.mu_lin3 = torch.nn.Linear(2500,2500, device=device)
        self.mu_ln0 = torch.nn.LayerNorm(2500, device=device)
        self.mu_ln1 = torch.nn.LayerNorm(2500, device=device)
        self.mu_ln2 = torch.nn.LayerNorm(2500, device=device)
        self.sigma_lin0 = torch.nn.Linear(1,2500, device=device)
        self.sigma_lin1 = torch.nn.Linear(2500,2500, device=device)
        self.sigma_lin2 = torch.nn.Linear(2500,2500, device=device)
        self.sigma_lin3 = torch.nn.Linear(2500,2500, device=device)
        self.sigma_ln0 = torch.nn.LayerNorm(2500, device=device)
        self.sigma_ln1 = torch.nn.LayerNorm(2500, device=device)
        self.sigma_ln2 = torch.nn.LayerNorm(2500, device=device)

        self.orb1 = torch.tensor(orbium, device=device, requires_grad=True)
        self.orb2 = torch.tensor(orbium, device=device, requires_grad=True).flip(0)

        self.mean = torch.zeros(batch_size, 50,50, requires_grad=False, device=device)
        self.std  = torch. ones(batch_size, 50,50, requires_grad=False, device=device)

    def forward(self):

        mu = self.mu_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        mu = self.mu_ln0(mu + torch.nn.functional.relu(self.mu_lin1(mu)))
        mu = self.mu_ln1(mu + torch.nn.functional.relu(self.mu_lin2(mu)))
        mu = self.mu_ln2(mu + torch.nn.functional.relu(self.mu_lin3(mu)))
        mu = mu.reshape(50,50)
        sigma = self.sigma_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        sigma = self.sigma_ln0(sigma + torch.nn.functional.relu(self.sigma_lin1(sigma)))
        sigma = self.sigma_ln1(sigma + torch.nn.functional.relu(self.sigma_lin2(sigma)))
        sigma = self.sigma_ln2(sigma + torch.nn.functional.relu(self.sigma_lin3(sigma)))
        sigma = sigma.reshape(50,50)

        x = torch.normal(mean=self.mean, std=self.std) * sigma + mu
        x = torch.clamp(x, 0, 1)
        x = torch.nn.functional.pad(x, (5,9,5,9,0,0))
        signal = mid_signal = running_signal = x

        loss = torch.zeros(signal.size(0)).to(signal.device);
        for i in range(1,self.iterations): 
            if i == self.iterations // 2: mid_signal = running_signal
            running_signal = self.lenia.run(running_signal)  
            loss += running_signal[:,:,:5].mean(2).mean(1)
            loss += running_signal[:,:,55:].mean(2).mean(1)
            loss -= running_signal[:,5:55,5:55].mean(2).mean(1)
            loss += running_signal[:,:5,5:55].mean(2).mean(1)
            loss += running_signal[:,55:,5:55].mean(2).mean(1)

        return loss, signal, mid_signal, running_signal

class Model3(torch.nn.Module):
    def __init__(self, batch_size = 10, iterations = 41, device="cpu"):
        super().__init__()
        self.device = device
        self.lenia = Lenia(size=64, device=device)
        self.iterations = iterations

        self.mu_lin0 = torch.nn.Linear(1,400, device=device)
        self.mu_lin1 = torch.nn.Linear(400,400, device=device)
        self.mu_lin2 = torch.nn.Linear(400,400, device=device)
        self.mu_lin3 = torch.nn.Linear(400,400, device=device)
        self.mu_lin4 = torch.nn.Linear(400,400, device=device)
        self.mu_ln0 = torch.nn.LayerNorm(400, device=device)
        self.mu_ln1 = torch.nn.LayerNorm(400, device=device)
        self.mu_ln2 = torch.nn.LayerNorm(400, device=device)
        self.mu_ln3 = torch.nn.LayerNorm(400, device=device)
        self.sigma_lin0 = torch.nn.Linear(1,400, device=device)
        self.sigma_lin1 = torch.nn.Linear(400,400, device=device)
        self.sigma_lin2 = torch.nn.Linear(400,400, device=device)
        self.sigma_lin3 = torch.nn.Linear(400,400, device=device)
        self.sigma_lin4 = torch.nn.Linear(400,400, device=device)
        self.sigma_ln0 = torch.nn.LayerNorm(400, device=device)
        self.sigma_ln1 = torch.nn.LayerNorm(400, device=device)
        self.sigma_ln2 = torch.nn.LayerNorm(400, device=device)
        self.sigma_ln3 = torch.nn.LayerNorm(400, device=device)

        self.orb1 = torch.tensor(orbium, device=device, requires_grad=True)
        self.orb2 = torch.tensor(orbium, device=device, requires_grad=True).flip(0)

        self.mean = torch.zeros(batch_size, 20,20, requires_grad=False, device=device)
        self.std  = torch. ones(batch_size, 20,20, requires_grad=False, device=device)

    def forward(self):

        mu = self.mu_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        mu = self.mu_ln0(mu + torch.nn.functional.relu(self.mu_lin1(mu)))
        mu = self.mu_ln1(mu + torch.nn.functional.relu(self.mu_lin2(mu)))
        mu = self.mu_ln2(mu + torch.nn.functional.relu(self.mu_lin3(mu)))
        mu = self.mu_ln3(mu + torch.nn.functional.relu(self.mu_lin4(mu)))
        mu = mu.reshape(20,20)
        sigma = self.sigma_lin0(torch.tensor([1], dtype=torch.float, device = self.device))
        sigma = self.sigma_ln0(sigma + torch.nn.functional.relu(self.sigma_lin1(sigma)))
        sigma = self.sigma_ln1(sigma + torch.nn.functional.relu(self.sigma_lin2(sigma)))
        sigma = self.sigma_ln2(sigma + torch.nn.functional.relu(self.sigma_lin3(sigma)))
        sigma = self.sigma_ln3(sigma + torch.nn.functional.relu(self.sigma_lin4(sigma)))
        sigma = sigma.reshape(20,20)

        x = torch.normal(mean=self.mean, std=self.std) * sigma + mu

        loss = torch.zeros(x.size(0), device=x.device)

        signal = torch.clamp(x, 0, 1)
        signal = torch.nn.functional.pad(signal, (0,44,0,44,0,0))
        running_signal = mid_signal = signal
        
        loss += ((x-1)**2).mean(1).mean(1)
        for i in range(1,self.iterations): 
            if i == self.iterations // 2: mid_signal = running_signal
            with torch.no_grad():
                running_signal = self.lenia.run(running_signal)  

            loss += ((running_signal[:,30:50,30:50] - 1)**2).mean(1).mean(1)
        #loss += ((running_signal[:,:20,:20] - 1)**2).mean(1).mean(1)
        #loss += running_signal[:,20:,:].mean(1).mean(1)
        #loss += running_signal[:,20:,20:].mean(1).mean(1)


        return loss, signal, mid_signal, running_signal



model = Model3(batch_size=16, iterations=51, device="cuda:0")
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

import beepy
for step in range(1000000):

    optimizer.zero_grad()

    loss, signal, mid_signal, running_signal = model()

    best:int = loss.argmin().item()
    loss = loss.mean()
    loss.backward()
    optimizer.step()


    print(f"{step}, {loss.item()}")
    if not step % 1000: 
        #beepy.beep(sound="ping")
        print(compile_cpp(signal[best]))
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(signal[best].detach().cpu().numpy(),vmin=0,vmax=1)
        axs[1].imshow(mid_signal[best].detach().cpu().numpy(),vmin=0,vmax=1)
        axs[2].imshow(running_signal[best].detach().cpu().numpy(),vmin=0,vmax=1)
        with open("/tmp/orbium.txt", "w") as f:
            f.write(str(running_signal[best].tolist()))
        plt.show()










