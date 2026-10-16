"""Kill-only Deadly Corridor; three skipped frames = four-tic action repeat."""
import itertools,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import vizdoom as vzd

BUTTONS=['MOVE_LEFT','MOVE_RIGHT','ATTACK','MOVE_FORWARD','MOVE_BACKWARD','TURN_LEFT','TURN_RIGHT']
# All combinations without opposing buttons: 3 strafe x 3 forward x 3 turn x 2 fire.
ACTIONS=np.asarray([[s==-1,s==1,fire,f==1,f==-1,t==-1,t==1]
    for s,f,t,fire in itertools.product((-1,0,1),(-1,0,1),(-1,0,1),(False,True))],dtype=np.int32)

class Corridor:
    def __init__(self,seed=202101,visible=False,action_repeat=4):
        if action_repeat!=4:raise ValueError('This prepared protocol skips three intermediate frames')
        self.repeat=action_repeat;self.game=vzd.DoomGame();p=Path(vzd.scenarios_path)
        self.game.load_config(str(p/'deadly_corridor.cfg'));self.game.set_doom_scenario_path(str(p/'deadly_corridor.wad'))
        self.game.set_window_visible(visible);self.game.set_sound_enabled(False);self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_render_hud(False);self.game.set_render_crosshair(False);self.game.set_render_weapon(False)
        self.game.set_doom_skill(5);self.game.set_episode_timeout(2100);self.game.set_seed(seed);self.game.init()
        actual=[b.name for b in self.game.get_available_buttons()]
        if actual!=BUTTONS:self.close();raise ValueError(f'Unexpected button order {actual}')
        self.provenance=dict(vizdoom_version=vzd.__version__,wad_sha256=hashlib.sha256((p/'deadly_corridor.wad').read_bytes()).hexdigest(),
            cfg_sha256=hashlib.sha256((p/'deadly_corridor.cfg').read_bytes()).hexdigest(),skill=5,timeout_tics=2100,
            action_repeat=4,skipped_frames=3,actions=len(ACTIONS),buttons=BUTTONS,observation='RGB only; area resize320x240 to100x100',
            overrides=dict(hud=False,crosshair=False,weapon=False,sound=False),reward='positive KILLCOUNT delta: +1 per kill, zero otherwise; native reward discarded')

    def observe(self):
        s=self.game.get_state()
        if s is None:return None
        x=torch.from_numpy(s.screen_buffer.copy()).permute(2,0,1).float()[None]/255
        # Store uint8 resized observations in replay; identical online/replay pixels.
        return (F.interpolate(x,(100,100),mode='area')[0]*255).round().byte().numpy()
    def reset(self,seed=None):
        if seed is not None:self.game.set_seed(int(seed))
        self.game.new_episode();return self.observe()
    def step(self,action):
        before=self.game.get_episode_time();kills_before=self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.make_action(ACTIONS[int(action)].tolist(),4)
        kills_after=self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        reward=max(0.,kills_after-kills_before)
        elapsed=self.game.get_episode_time()-before;done=self.game.is_episode_finished()
        # The scenario's 2100-tic limit is the task's finite horizon, not an
        # external training truncation. It and death both end this episodic return.
        info=dict(kills=int(kills_after),kill_delta=reward,tics=elapsed,dead=self.game.is_player_dead(),episode_tics=self.game.get_episode_time())
        return self.observe(),float(reward),bool(done),info
    def close(self):self.game.close()
