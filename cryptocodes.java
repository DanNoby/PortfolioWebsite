
// Crypto codes

////////////  RSA  //////////// 
import java.util.*;
import java.math.*;


public class RSA{
    public static void main(String[] args){
        int nInt = 143;
        int eInt = 7;
        int pInt = 0, qInt = 0;
        for (int i=2;i<=Math.sqrt(nInt);i++){
            if (nInt%i==0){
                pInt = i;
                qInt = nInt/i;
                break;
            }
        }
        System.out.println("P = "+pInt);
        System.out.println("Q = "+qInt);

        BigInteger n = BigInteger.valueOf(nInt);
        BigInteger e = BigInteger.valueOf(eInt);
        BigInteger p = BigInteger.valueOf(pInt);
        BigInteger q = BigInteger.valueOf(qInt);

        BigInteger phi = p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));

        BigInteger msg = BigInteger.valueOf(64);

        BigInteger d = e.modInverse(phi);

        BigInteger cipher = msg.modPow(e,n);

        BigInteger decrypted = cipher.modPow(d,n);

        System.out.println("phi = "+phi);
        System.out.println("Msg = "+msg);
        System.out.println("Cipher = "+cipher);
        System.out.println("Decrypted = "+decrypted);
    }
}


/////////////// MITM ////////// 

import java.util.*;
import java.math.*;

public class MITM{
    public static void main(String[] args){
        BigInteger XA = BigInteger.valueOf(9);//Alice priv
        BigInteger XB = BigInteger.valueOf(14);//Bob priv

        System.out.println("Alice's Priv Key = " +XA);
        System.out.println("Bob's Priv Key = " + XB);

        BigInteger q = BigInteger.valueOf(29);
        BigInteger a = BigInteger.valueOf(2);

        BigInteger XM1 = BigInteger.valueOf(5); //with bob
        BigInteger XM2 = BigInteger.valueOf(11); //with alice

        System.out.println("Middle man's Priv Key(Bob) = " +XM1);
        System.out.println("Middle man's Priv Key(Alice) = " +XM2);


        // computing public keys

        BigInteger YA = a.modPow(XA,q);
        BigInteger YB = a.modPow(XB,q);

        System.out.println("Alice's Pub Key = " +YA);
        System.out.println("Bob's Pub Key = " +YB);


        //middle man's public keys

        BigInteger YM1 = a.modPow(XM1,q);
        BigInteger YM2 = a.modPow(XM2,q);

        System.out.println("MM's Pub Key(Bob) = " +YM1);
        System.out.println("MM's Pub Key(Alice) = " +YM2);

        //Alice n Bob's Secret Key
        BigInteger K = YA.modPow(XB,q);

        System.out.println("Alice & Bob's Secret Key = " +K);

        //Alice n Middleman's Secret Key
        BigInteger K2 = YM2.modPow(XA,q);
        System.out.println("Alice & MM's Secret Key = " +K2);


        //Bob n Middleman's Secret Key
        BigInteger K1 = YM1.modPow(XB,q);

        System.out.println("Bob & MM's Secret Key = " +K1);

    }
}

/////////   SDES   /////////

import java.util.*;

public class SDES {

    static int[] P10={3,5,2,7,4,10,1,9,8,6};
    static int[] P8 ={6,3,7,4,8,5,10,9};
    static int[] IP ={2,6,3,1,4,8,5,7};
    static int[] IP1={4,1,3,5,7,2,8,6};
    static int[] EP ={4,1,2,3,2,3,4,1};
    static int[] P4 ={2,4,3,1};

    static int[][] S0={{1,0,3,2},{3,2,1,0},{0,2,1,3},{3,1,3,2}};
    static int[][] S1={{0,1,2,3},{2,0,1,3},{3,0,1,0},{2,1,0,3}};

    static String perm(String s,int[] p){
        String r="";
        for(int i:p) r+=s.charAt(i-1);
        return r;
    }

    static String ls(String s,int n){
        return s.substring(n)+s.substring(0,n);
    }

    static String xor(String a,String b){
        String r="";
        for(int i=0;i<a.length();i++)
            r+= (a.charAt(i)==b.charAt(i))?'0':'1';
        return r;
    }

    static String sbox(String s,int[][] box){
        int row=Integer.parseInt(""+s.charAt(0)+s.charAt(3),2);
        int col=Integer.parseInt(""+s.charAt(1)+s.charAt(2),2);
        return String.format("%2s",
            Integer.toBinaryString(box[row][col])).replace(' ','0');
    }

    static String[] keys(String key){
        key=perm(key,P10);
        String l=ls(key.substring(0,5),1);
        String r=ls(key.substring(5),1);
        String k1=perm(l+r,P8);
        l=ls(l,2); r=ls(r,2);
        String k2=perm(l+r,P8);
        return new String[]{k1,k2};
    }

    static String fk(String s,String k){
        String l=s.substring(0,4);
        String r=s.substring(4);
        String t=perm(r,EP);
        t=xor(t,k);
        String s0=sbox(t.substring(0,4),S0);
        String s1=sbox(t.substring(4),S1);
        String p=perm(s0+s1,P4);
        return xor(l,p)+r;
    }

    static String encrypt(String pt,String key){
        String[] k=keys(key);
        pt=perm(pt,IP);
        pt=fk(pt,k[0]);
        pt=pt.substring(4)+pt.substring(0,4);
        pt=fk(pt,k[1]);
        return perm(pt,IP1);
    }

    static String decrypt(String ct,String key){
        String[] k=keys(key);
        ct=perm(ct,IP);
        ct=fk(ct,k[1]);
        ct=ct.substring(4)+ct.substring(0,4);
        ct=fk(ct,k[0]);
        return perm(ct,IP1);
    }

    public static void main(String[] args){
        String key="1010000010";
        String pt="11010111";

        String ct=encrypt(pt,key);
        System.out.println("Cipher: "+ct);
        System.out.println("Decrypted: "+decrypt(ct,key));
    }
}


///////  AES  /////////////

public class AES {

    // SHIFT ROWS
    static void shiftRows(byte[][] state) {
        for (int r = 1; r < 4; r++) {
            byte[] temp = new byte[4];

            for (int c = 0; c < 4; c++) {
                temp[c] = state[r][(c + r) % 4];
            }

            for (int c = 0; c < 4; c++) {
                state[r][c] = temp[c];
            }
        }
    }

    // multiply by 2 in GF(2^8)
    static byte xtime(byte b) {
    int x = b & 0xFF;
    x <<= 1;

    if ((b & 0x80) != 0) {
        x ^= 0x1B;
    }

    return (byte)(x & 0xFF);
}

    // general multiplication in GF(2^8)
    static byte gmul2(byte b) {
        return xtime(b);
    }

    static byte gmul3(byte b) {
        return (byte)(xtime(b) ^ b);
    }

    // MIX COLUMNS
    static void mixColumns(byte[][] state) {

        for (int c = 0; c < 4; c++) {

            byte s0 = state[0][c];
            byte s1 = state[1][c];
            byte s2 = state[2][c];
            byte s3 = state[3][c];

            state[0][c] = (byte)(gmul2(s0) ^ gmul3(s1) ^ s2 ^ s3);
            state[1][c] = (byte)(s0 ^ gmul2(s1) ^ gmul3(s2) ^ s3);
            state[2][c] = (byte)(s0 ^ s1 ^ gmul2(s2) ^ gmul3(s3));
            state[3][c] = (byte)(gmul3(s0) ^ s1 ^ s2 ^ gmul2(s3));
        }
    }

    // print matrix
    static void printState(byte[][] state) {

        for (int r = 0; r < 4; r++) {

            for (int c = 0; c < 4; c++) {
                System.out.printf("%02X ", state[r][c]);
            }

            System.out.println();
        }

        System.out.println();
    }

    public static void main(String[] args) {

        byte[][] state = {
            {(byte)0x87,(byte)0xF2,(byte)0x4D,(byte)0x97},
            {(byte)0xEC,(byte)0x6E,(byte)0x4C,(byte)0x90},
            {(byte)0x4A,(byte)0xC3,(byte)0x46,(byte)0xE7},
            {(byte)0x8C,(byte)0xD8,(byte)0x95,(byte)0xA6}
        };

        System.out.println("Initial State:");
        printState(state);

        shiftRows(state);
        System.out.println("After ShiftRows:");
        printState(state);

        mixColumns(state);
        System.out.println("After MixColumns:");
        printState(state);
    }
}



// client-server code incase , also readLong() and writeLong() exists for BigInteger transfer


// sender
import java.io.*;
import java.net.*;

public class SenderClient {
    public static void main(String[] args) throws Exception {
        Socket socket = new Socket("localhost", 5000);
        DataOutputStream output = new DataOutputStream(socket.getOutputStream());

        String message = "HELLO";
        
        // --- ENCRYPTION LOGIC HERE ---
        String cipherText = message; // Replace with your encrypt function
        
        output.writeUTF(cipherText);
        System.out.println("Message sent: " + cipherText);

        socket.close();
    }
}

// reciever
import java.io.*;
import java.net.*;

public class ReceiverServer {
    public static void main(String[] args) throws Exception {
        ServerSocket server = new ServerSocket(5000);
        System.out.println("Receiver is waiting...");
        
        Socket socket = server.accept();
        DataInputStream input = new DataInputStream(socket.getInputStream());

        String cipherText = input.readUTF();
        System.out.println("Received (Cipher): " + cipherText);

        // --- DECRYPTION LOGIC HERE ---
        String plainText = cipherText; // Replace with your decrypt function
        System.out.println("Decrypted: " + plainText);

        socket.close();
        server.close();
    }
}
