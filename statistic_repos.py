import os
import config

working_dir = os.path.dirname(os.path.realpath(__file__))

for repo in config.repos:
    if not repo['active']:
        continue

    repo_path = os.path.join(config.repo_dir, repo['name'])

    # clone repo if not exists
    if not os.path.exists(repo_path):
        command = 'git clone -b %s %s %s' % (repo['branch'], repo['url'], repo_path)
        print 'clone repo: %s' % command
        os.system(command)

    # update repo
    command = 'cd %s && git pull && cd %s' % (repo_path, working_dir)
    print 'update repo: ', command
    os.system(command)

    # generate statistic html
    statistic_dir = os.path.join(config.vis_dir, repo['name'])
    if not os.path.exists(statistic_dir):
        os.makedirs(statistic_dir)

    command = './gitstats/gitstats %s %s' % (repo_path, statistic_dir)
    print command
    os.system(command)
