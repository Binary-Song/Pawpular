
$hostip = $args[0]
$port = $args[1]
$user = $args[2]

if (!$hostip) {
    write-host "Usage: copy-key <host> <port> <user>"
    exit
}

cat ~/.ssh/id_rsa.pub | ssh $user@$hostip -p $port "echo '--- copying ssh key ---';mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; cat ~/.ssh/authorized_keys"