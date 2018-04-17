from src.env.LOLEnv import LOLEnv
from src.utils.training_utils import *
from src.utils.game_utils import *
from src.models.models import mlp
from src.models.learn import DeepQNetwork
from src.constants.Defines import *
from src.constants.config import *
from src.players.defensive_ai import LOLAI

import time
import queue
import tensorflow as tf
import threading as td


def play(ip, port, self_ai, parent_ai, data_queue, score_queue):

    env = LOLEnv(ip, port)
    rule = LOLAI(CAMP_BLUE)
    explorer = LOLAI(CAMP_RED)

    obs = Observation()
    history = History(HISTORY_LENGTH)
    episode = 0

    while True:
            # ========================================== start a new episode ==========================================
            frame_counter = 0  # record the num of frames which are stored, not include the skipped ones
            synchronize_version(self_ai, parent_ai)
            obs.build(env.reset()[1])
            episode += 1
            time_start = time.time()

            hero_sars = [SARS(history, data_queue, CAMP_RED, 0), SARS(history, data_queue, CAMP_RED, 1)]
            score = 0  # accumulated reward along the whole episode added from two players
            blue_hero_i = 0
            red_hero_i = 0

            while True:
                # ======================================= select ai and act ===========================================
                ai_camp = frame_counter % 2
                winner = obs.judge_winner()

                if ai_camp == CAMP_BLUE:
                    if winner is None:
                        # select rule-based AI as player and make decision
                        ai = rule
                        # take action
                        obs.consume_frame(SKIP_FRAME, env, ai.act(obs, blue_hero_i))
                        # renew the hero index
                        blue_hero_i = (blue_hero_i + 1) % 2
                    else:
                        # when the game is end, shape reward sand save trajectory for both NN players
                        score += hero_sars[0].end(obs)
                        score += hero_sars[1].end(obs)

                else:  # CAMP_RED
                    # put s-a-r-s' trajectory into history
                    score += hero_sars[red_hero_i].end(obs)

                    if winner is None:
                        # select NN as player and make decision
                        ai = self_ai
                        # hero_sars[red_hero_i].s1 is now the newest extracted feature
                        output = ai.act(hero_sars[red_hero_i].s1, obs, red_hero_i, port, explorer)
                        # save action
                        hero_sars[red_hero_i].get_action(output)
                        # take action
                        obs.consume_frame(SKIP_FRAME, env, Action.wrap_action(output, obs, CAMP_RED, red_hero_i))
                        # renew the hero index
                        red_hero_i = (red_hero_i + 1) % 2
                    else:
                        # when the game is end, shape reward sand save trajectory for the other player as well
                        score += hero_sars[1 - red_hero_i].end(obs)

                # ======================================= when the game ends ==========================================
                if winner is not None or frame_counter >= MAX_FRAMES_PER_EPISODE:
                    print()
                    # ============= time printer =============
                    time_cost = time.time() - time_start
                    print("Port {}'s episode {} ends, time : {:.2f}s, every player has {:.2f} frames in 1s."
                          .format(port, episode, time_cost, frame_counter / 4. / time_cost))
                    # ============ winner printer ============
                    if winner is not None:
                        if winner == CAMP_RED:
                            print('CAMP_RED win!')
                        elif winner == CAMP_BLUE:
                            print('CAMP_BLUE win!')
                    else:
                        print('No winner...')
                    # ============ score handler =============
                    print("Score: {:2f}".format(score))
                    if port % 5 == 0:
                        score_queue.put(score)
                    print("Epsilon : %f" % self_ai.epsilon)
                    print()
                    env.end()
                    break

                # ========================================== count frames =============================================
                frame_counter += 1


if __name__ == '__main__':

    Sess = tf.Session()
    global_ai = DeepQNetwork(
        n_features=QUANTITY_FEATURES,
        n_actions=QUANTITY_ACTIONS,
        scope='global_ai',
        model=mlp,
        parent_ai=None,
        sess=Sess,
        learning_rate=5e-3,
        n_replace_target=50,
        hiddens=HIDDENS,
        decay=0.99,
        memory_size=1000000,
        batch_size=3000,
        epsilon_decrement=5e-4,
        epsilon_lower=0.2,
        learn_start=LEARN_START,
    )

    dataQ = queue.Queue()

    score_plotter = ScorePlotter()
    scoreQ = queue.Queue()

    ais = []
    for i in range(N_GAMES):
        ais.append(
            DeepQNetwork(
                n_features=QUANTITY_FEATURES,
                n_actions=QUANTITY_ACTIONS,
                scope='local_ai_' + str(i),
                model=mlp,
                parent_ai=global_ai,
                sess=Sess,
                hiddens=HIDDENS
            )
        )

    Saver = tf.train.Saver()

    if RESTORE:
        Saver.restore(Sess, RESTORE_PATH)
        if RESET_EPSILON:
            Sess.run(global_ai.reset_epsilon)
        print('restored successfully from ' + RESTORE_PATH)
    else:
        Sess.run(tf.global_variables_initializer())

    for i in range(N_GAMES):
        new_thread = td.Thread(target=play, args=(IP, PORT[i], ais[i], global_ai, dataQ, scoreQ))
        new_thread.start()

    while LEARNING:
        plot_score(score_plotter, scoreQ)
        fetch_data(global_ai, dataQ)
        global_ai.learn()
        if not ONLY_PLAY and global_ai.learn_step % SAVE_EVERY == 1:
            save_path = Saver.save(Sess, SAVE_PATH + str(global_ai.learn_step) + '.ckpt')
            print('saved in' + save_path)
