cat > ~/backup_qaoa.sh <<'EOF'
#!/bin/bash
mkdir -p $SCRATCH/dhanvib_backups
cp -r ~/QAOA $SCRATCH/dhanvib_backups/QAOA_backup_$(date +%Y-%m-%d)
chmod -R 700 $SCRATCH/dhanvib_backups/QAOA_backup_$(date +%Y-%m-%d)
echo "Backup complete: $SCRATCH/dhanvib_backups/QAOA_backup_$(date +%Y-%m-%d)"
EOF

chmod +x ~/backup_qaoa.sh
